"""
Local answer generation using Ollama.
The retrieval stage runs independently; this module only activates
when the top semantic score passes the gate threshold.
"""
import json
import re
import time

from app.settings import OLLAMA_MODEL, OLLAMA_BASE_URL

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research assistant specialised in linguistic revitalisation of \
indigenous languages. You answer questions strictly based on the provided \
source fragment. You MUST respond in the same language as the question: \
if the question is in Spanish, answer in Spanish; if in English, answer in English.

Rules:
1. If the fragment contains enough information, answer concisely (2-4 sentences).
2. If the fragment does not contain enough information, set "abstain" to true.
3. Always include at least one verbatim quote of 20+ characters copied exactly \
from the text that follows the fragment header. The quote must be a substring \
of the source text, not the header line.
4. Use the exact chunk_id shown in brackets at the start of the fragment header, \
for example if the header is "[D181-S23] Title" then chunk_id is "D181-S23".
5. Return ONLY a JSON object with this exact structure:
{{
  "abstain": false,
  "answer": "tu respuesta aquí en el mismo idioma que la pregunta",
  "evidence": [
    {{"chunk_id": "D181-S23", "quote": "verbatim excerpt from the source text"}}
  ]
}}
Do not add any text outside the JSON object.\
"""

REFUSAL = (
    "El corpus no contiene información suficiente para responder esta consulta. "
    "Intenta reformularla o consulta directamente las fuentes del corpus."
)


# ── Message builder ───────────────────────────────────────────────────────────

def build_messages(query: str, sources: list[dict]) -> list[dict]:
    """
    Keep source content in the user message (data), not in the system prompt
    (instructions). This reduces prompt-injection risk from document content.
    Includes a suggested quote to help small models cite correctly.
    """
    source = sources[0]
    suggested_quote = _extract_best_quote(source["text"]) or source["text"][:80]
    source_text = f"[{source['chunk_id']}] {source['title']} \u2014 {source['heading']}\n{source['text']}"
    user_content = (
        f"SOURCE FRAGMENT:\n{source_text}\n\n"
        f"SUGGESTED QUOTE (copy this verbatim into evidence.quote): \"{suggested_quote}\"\n\n"
        f"QUESTION (answer in Spanish): {query}"
    )
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


# ── Response validator ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Collapse whitespace, normalise dashes and lowercase for substring matching."""
    text = " ".join(text.split())
    text = re.sub(r"[\u2013\u2014\u2012\u2011]", "-", text)
    return text.lower()


def _extract_best_quote(source_text: str, min_len: int = 20) -> str | None:
    """Return the first sentence of source_text that is long enough to be a quote."""
    for sent in re.split(r"[.;]", source_text):
        sent = sent.strip()
        if len(sent) >= min_len:
            return sent
    return None


def validate_generation(raw_output: str, sources: list[dict]) -> dict:
    """
    Parse and verify the model's JSON output.
    Returns a safe result dict; never exposes unverified drafts.
    """
    failure = {"status": "verification_failed", "answer": REFUSAL,
               "evidence": [], "raw_output": raw_output}

    # Extract JSON even if the model wraps it in markdown fences
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return failure

    try:
        draft = json.loads(match.group())
    except json.JSONDecodeError:
        return failure

    if not isinstance(draft, dict):
        return failure

    # Explicit abstention
    if draft.get("abstain") is True:
        return {"status": "abstained", "answer": REFUSAL,
                "evidence": [], "raw_output": raw_output}

    if "abstain" not in draft:
        return failure

    answer = draft.get("answer", "")
    if not isinstance(answer, str) or len(answer.strip()) < 10:
        return failure

    evidence_list = draft.get("evidence", [])
    if not isinstance(evidence_list, list) or not evidence_list:
        return failure

    # Build lookup: chunk_id → normalised text (only from passed sources)
    source_map   = {s["chunk_id"]: s["text"] for s in sources}
    source_texts = {k: _normalise(v) for k, v in source_map.items()}

    verified_evidence = []
    for ev in evidence_list:
        chunk_id = ev.get("chunk_id", "")
        quote    = ev.get("quote", "")

        if chunk_id not in source_texts:
            return failure

        if not isinstance(quote, str) or len(quote.strip()) < 20:
            return failure

        if _normalise(quote) not in source_texts[chunk_id]:
            return failure

        verified_evidence.append({"chunk_id": chunk_id, "quote": quote.strip()})

    if not verified_evidence:
        return failure

    return {
        "status":   "answered",
        "answer":   answer.strip(),
        "evidence": verified_evidence,
        "raw_output": raw_output,   # stripped before sending to client
    }


# ── Ollama generator ──────────────────────────────────────────────────────────

class LocalGenerator:
    def __init__(self) -> None:
        try:
            import ollama
            self._client = ollama.Client(host=OLLAMA_BASE_URL)
            models = [m.model for m in self._client.list().models]
            if not any(OLLAMA_MODEL in m for m in models):
                raise RuntimeError(
                    f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                    f"Run: ollama pull {OLLAMA_MODEL}"
                )
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. "
                "Run: pip install ollama"
            ) from e

    def generate(self, query: str, sources: list[dict]) -> dict:
        """
        Generate an answer grounded in the top retrieved source.
        Only the first source is passed to keep the answer focused.
        """
        messages = build_messages(query, sources[:1])
        start    = time.perf_counter()

        try:
            response = self._client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0, "num_predict": 512},
            )
            raw = response.message.content
        except Exception as e:
            return {
                "status":   "generation_error",
                "answer":   REFUSAL,
                "evidence": [],
                "generation_seconds": round(time.perf_counter() - start, 3),
            }

        result = validate_generation(raw, sources[:1])
        result["generation_seconds"] = round(time.perf_counter() - start, 3)
        return result
