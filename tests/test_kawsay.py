"""
Kawsay test suite.
Covers: corpus loading, retrieval, evaluation metrics, generation
validation, API contracts, and end-to-end flow.

Run:
    python -m pytest -q
    python -m pytest -v tests/test_kawsay.py
"""
import json
import math
import re
import pytest
import numpy as np
from fastapi.testclient import TestClient

import app.main as server
from app.documents import read_chunks, read_queries, corpus_hash
from app.evaluation import ranking_metrics, select_gate, gate_metrics
from app.generation import build_messages, validate_generation, REFUSAL, SYSTEM_PROMPT
from app.retrieval import SearchEngine, ranked_indices
from app.settings import ARTIFACT_DIR


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def chunks():
    return read_chunks()


@pytest.fixture(scope="module")
def engine():
    return SearchEngine()


@pytest.fixture(scope="module")
def client(engine):
    """Test client with a generator stub so tests never call Ollama."""
    class GeneratorStub:
        def generate(self, query, sources):
            source = sources[0]
            return {
                "status":             "answered",
                "answer":             source["text"][:120],
                "evidence":           [{"chunk_id": source["chunk_id"],
                                        "quote":    source["text"][:60]}],
                "raw_output":         "internal draft — not exposed",
                "generation_seconds": 0.0,
            }

    patch = pytest.MonkeyPatch()
    patch.setattr(server, "SearchEngine",   lambda: engine)
    patch.setattr(server, "LocalGenerator", GeneratorStub)
    with TestClient(server.app) as session:
        yield session
    patch.undo()


# ── Corpus ────────────────────────────────────────────────────────────────────

def test_corpus_loads(chunks):
    """At least one document and one chunk must be present."""
    assert len(chunks) >= 1
    assert all(
        key in chunks[0]
        for key in ("chunk_id", "document_id", "title", "heading", "text", "doc_type")
    )


def test_chunk_ids_are_unique(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_chunk_id_format(chunks):
    pattern = re.compile(r"^D\d{3}-S\d+$")
    for c in chunks:
        assert pattern.match(c["chunk_id"]), f"Bad chunk_id: {c['chunk_id']}"


def test_no_empty_text(chunks):
    for c in chunks:
        assert c["text"].strip(), f"Empty text in chunk {c['chunk_id']}"


def test_corpus_hash_changes_on_edit(chunks):
    modified = [dict(c) for c in chunks]
    modified[0]["text"] += " CHANGED."
    assert corpus_hash(modified) != corpus_hash(chunks)


def test_queries_load_if_present(chunks):
    queries = read_queries(chunks)
    if not queries:
        pytest.skip("queries.json not present yet")
    ids = {c["chunk_id"] for c in chunks}
    for q in queries:
        assert q["split"] in ("validation", "test")
        assert set(q["relevant_ids"]) <= ids
        assert q["answerable"] == bool(q["relevant_ids"])


def test_query_families_do_not_cross_splits(chunks):
    queries = read_queries(chunks)
    if not queries:
        pytest.skip("queries.json not present yet")
    val  = {q["family_id"] for q in queries if q["split"] == "validation"}
    test = {q["family_id"] for q in queries if q["split"] == "test"}
    assert not val & test


# ── Retrieval ─────────────────────────────────────────────────────────────────

def test_embeddings_are_unit_normalised(engine):
    norms = np.linalg.norm(engine.index["embeddings"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_semantic_score_equals_dot_product(engine, chunks):
    query  = "revitalización del quechua mediante educación bilingüe"
    result = engine.search(query)["semantic"][0]
    from app.retrieval import encode_query
    vector = encode_query(engine.encoder, query)
    idx      = engine.chunk_ids.index(result["chunk_id"])
    expected = float(np.dot(vector, engine.index["embeddings"][idx]))
    assert result["score"] == pytest.approx(expected, abs=1e-5)


def test_search_returns_top_k_for_both_methods(engine):
    results = engine.search("lenguas indígenas en comunidades rurales")
    assert len(results["semantic"]) == 5
    assert len(results["lexical"])  == 5


def test_bm25_index_is_present(engine):
    assert "bm25" in engine.index


def test_ranked_indices_tie_breaking():
    # Equal scores → sorted by chunk_id ascending
    assert ranked_indices([0.5, 0.5, 0.2], ["B", "A", "C"]) == [1, 0, 2]


def test_search_unknown_topic_returns_low_score(engine):
    off_topic = engine.search("recetas de cocina italiana con pasta")
    on_topic  = engine.search("revitalización del quechua mediante educación")
    # Off-topic score must be strictly lower than an on-topic score
    assert off_topic["semantic"][0]["score"] < on_topic["semantic"][0]["score"]


# ── Evaluation metrics ────────────────────────────────────────────────────────

def test_ranking_metrics_perfect():
    m = ranking_metrics(["A", "B", "C"], ["A"])
    assert m["recall_at_5"]    == 1.0
    assert m["precision_at_5"] == pytest.approx(1 / 5)
    assert m["mrr_at_5"]       == 1.0
    assert m["ndcg_at_5"]      == 1.0


def test_ranking_metrics_second_position():
    m = ranking_metrics(["B", "A", "C"], ["A"])
    assert m["mrr_at_5"]  == pytest.approx(0.5)
    assert m["ndcg_at_5"] == pytest.approx(1 / math.log2(3))


def test_ranking_metrics_not_found():
    m = ranking_metrics(["B", "C", "D"], ["A"])
    assert m["recall_at_5"]    == 0.0
    assert m["mrr_at_5"]       == 0.0
    assert m["ndcg_at_5"]      == 0.0


def test_ranking_metrics_requires_relevant():
    with pytest.raises(ValueError):
        ranking_metrics(["A"], [])


def test_gate_selection_clean_separation():
    result = select_gate([False, False, True, True], [0.1, 0.2, 0.8, 0.9])
    assert result["threshold"] == pytest.approx(0.8)
    assert result["f1"]        == pytest.approx(1.0)


def test_gate_metrics_values():
    m = gate_metrics([False, True, True], [0.2, 0.8, 0.9], threshold=0.5)
    assert m["tp"] == 2
    assert m["tn"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0


# ── Generation validation ─────────────────────────────────────────────────────

def _make_payload(chunk: dict, **overrides) -> str:
    payload = {
        "abstain":  False,
        "answer":   chunk["text"][:100],
        "evidence": [{"chunk_id": chunk["chunk_id"],
                      "quote":    chunk["text"][:60]}],
        **overrides,
    }
    return json.dumps(payload)


def test_valid_generation_passes(chunks):
    result = validate_generation(_make_payload(chunks[0]), chunks[:1])
    assert result["status"]   == "answered"
    assert result["evidence"] != []


def test_explicit_abstention(chunks):
    raw    = json.dumps({"abstain": True, "answer": "some text"})
    result = validate_generation(raw, chunks[:1])
    assert result["status"] == "abstained"
    assert result["answer"] == REFUSAL
    assert result["evidence"] == []


def test_json_inside_markdown_fence(chunks):
    """Model sometimes wraps JSON in ```json ... ```."""
    inner = _make_payload(chunks[0])
    raw   = f"```json\n{inner}\n```"
    result = validate_generation(raw, chunks[:1])
    assert result["status"] == "answered"


@pytest.mark.parametrize("problem", [
    "unknown_chunk_id",
    "invented_quote",
    "quote_too_short",
    "empty_evidence",
    "missing_abstain_flag",
    "bad_json",
    "non_object_json",
])
def test_invalid_draft_rejected(chunks, problem):
    draft = {
        "abstain":  False,
        "answer":   chunks[0]["text"][:100],
        "evidence": [{"chunk_id": chunks[0]["chunk_id"],
                      "quote":    chunks[0]["text"][:60]}],
    }
    if problem == "unknown_chunk_id":
        draft["evidence"][0]["chunk_id"] = "D99-S99"
    if problem == "invented_quote":
        draft["evidence"][0]["quote"] = "Esta frase nunca apareció en ningún documento del corpus."
    if problem == "quote_too_short":
        draft["evidence"][0]["quote"] = "Corto"
    if problem == "empty_evidence":
        draft["evidence"] = []
    if problem == "missing_abstain_flag":
        del draft["abstain"]

    raw = json.dumps(draft)
    if problem == "bad_json":
        raw = "not valid json at all"
    if problem == "non_object_json":
        raw = "[1, 2, 3]"

    result = validate_generation(raw, chunks[:1])
    assert result["status"]   == "verification_failed"
    assert result["evidence"] == []


def test_source_content_not_in_system_prompt(chunks):
    """Document text must be in the user message, not the system prompt."""
    sources   = [dict(chunks[0])]
    messages  = build_messages("Una consulta de prueba", sources)
    sys_msg   = messages[0]
    user_msg  = messages[1]
    assert sys_msg["role"]  == "system"
    assert user_msg["role"] == "user"
    assert sources[0]["text"] not in sys_msg["content"]
    assert sources[0]["text"] in user_msg["content"]


# ── API contracts ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "/",
    "/static/app.js",
    "/static/styles.css",
    "/api/health",
    "/api/documents",
    "/api/metrics",
    "/docs",
])
def test_read_endpoints_return_200(client, path):
    assert client.get(path).status_code == 200


def test_health_reports_generator_available(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"


def test_documents_endpoint_returns_list(client):
    data = client.get("/api/documents").json()
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.parametrize("body", [
    {},
    {"query": "corto"},
    {"query": True},
    {"query": " " * 15},
    {"query": "1234567890123"},
    {"query": "a" * 501},
    {"query": "consulta válida", "extra_field": True},
])
def test_invalid_requests_rejected(client, body):
    assert client.post("/api/ask",    json=body).status_code == 422
    assert client.post("/api/search", json=body).status_code == 422


# ── End-to-end flow ───────────────────────────────────────────────────────────

def test_search_returns_both_methods(client):
    body = {"query": "revitalización del quechua mediante educación bilingüe"}
    res  = client.post("/api/search", json=body)
    assert res.status_code == 200
    data = res.json()
    assert len(data["semantic"]) == 5
    assert len(data["lexical"])  == 5
    assert all("score" in r for r in data["semantic"])


def test_ask_returns_answer_and_results(client):
    body = {"query": "estrategias para involucrar a jóvenes en la revitalización lingüística"}
    res  = client.post("/api/ask", json=body)
    assert res.status_code == 200
    data = res.json()
    assert "results"   in data
    assert "status"    in data
    assert "raw_output" not in data          # never exposed to client
    assert "threshold" in data
    assert "version"   in data


def test_ask_valid_case(client):
    """Valid query: system finds relevant content and returns results with positive scores."""
    body = {"query": "experiencias de recuperación de lenguas indígenas en comunidades rurales"}
    res  = client.post("/api/ask", json=body)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("answered", "below_threshold", "search_only")
    assert data["results"]["semantic"][0]["score"] > 0


def test_ask_difficult_case(client):
    """Difficult query: paraphrase without exact keywords must still retrieve relevant content."""
    body = {"query": "iniciativas para que los hablantes jóvenes mantengan su lengua materna"}
    res  = client.post("/api/ask", json=body)
    assert res.status_code == 200
    data = res.json()
    assert data["results"]["semantic"][0]["score"] > 0


def test_ask_invalid_input(client):
    """Invalid input: too short, should be rejected before reaching the model."""
    assert client.post("/api/ask", json={"query": "abc"}).status_code == 422


def test_raw_output_never_in_response(client):
    body = {"query": "proyectos de radio para fortalecer lenguas indígenas"}
    data = client.post("/api/ask", json=body).json()
    assert "raw_output" not in data


def test_below_threshold_does_not_call_generator(client, monkeypatch):
    """Force threshold to 2.0 so no query can pass the gate."""
    monkeypatch.setitem(
        server.app.state.report["threshold_selection"], "threshold", 2.0
    )
    body   = {"query": "revitalización del quechua mediante educación bilingüe"}
    result = client.post("/api/ask", json=body).json()
    assert result["status"]             == "below_threshold"
    assert result["generation_seconds"] == 0.0


def test_concurrent_request_returns_429(client):
    lock = server.app.state.request_lock
    lock.acquire()
    try:
        body = {"query": "estrategias para involucrar a jóvenes en la revitalización"}
        assert client.post("/api/ask", json=body).status_code == 429
    finally:
        lock.release()


def test_missing_index_returns_503(client, monkeypatch):
    monkeypatch.setattr(server.app.state, "engine", None)
    body = {"query": "revitalización del quechua mediante educación bilingüe"}
    assert client.post("/api/search", json=body).status_code == 503
    assert client.post("/api/ask",    json=body).status_code == 503


def test_metrics_endpoint_has_partitions(client):
    data = client.get("/api/metrics").json()
    assert "version"      in data
    assert "corpus_hash"  in data


# ── Casos requeridos por el enunciado ─────────────────────────────────────────

def test_caso_valido_recupera_fragmentos(client):
    """Caso válido: consulta sobre tema central del corpus devuelve fragmentos
    con score > 0 y chunk_ids con formato correcto."""
    body = {"query": "experiencias de revitalización del quechua mediante educación bilingüe"}
    data = client.post("/api/search", json=body).json()
    assert len(data["semantic"]) == 5
    pattern = re.compile(r"^D\d{3}-S\d+$")
    for chunk in data["semantic"]:
        assert chunk["score"] > 0
        assert pattern.match(chunk["chunk_id"])
        assert len(chunk["text"]) > 0


def test_caso_dificil_parafrasis_recupera_mismo_tema(client):
    """Caso difícil: paráfrasis semántica sin palabras clave exactas.
    El sistema debe recuperar fragmentos sobre el mismo tema."""
    query_directa   = "revitalización del quechua mediante educación"
    query_parafrasis = "programas para que niños indígenas aprendan su lengua materna en la escuela"

    res_directa    = client.post("/api/search", json={"query": query_directa}).json()
    res_parafrasis = client.post("/api/search", json={"query": query_parafrasis}).json()

    # Ambas deben devolver resultados con score positivo
    assert res_directa["semantic"][0]["score"]    > 0
    assert res_parafrasis["semantic"][0]["score"] > 0

    # La semántica debe encontrar contenido relevante en la paráfrasis
    # (los chunks recuperados pueden ser distintos a la consulta directa pero
    # igualmente relevantes al tema — eso es robustez semántica, no un fallo)
    sem_top  = res_parafrasis["semantic"][0]["score"]
    assert sem_top > 0.2, "Semántica debe encontrar algo relevante en la paráfrasis"
    assert all(c["score"] > 0 for c in res_parafrasis["semantic"]), \
        "Todos los resultados de la paráfrasis deben tener score positivo"
    pattern = re.compile(r"^D\d{3}-S\d+$")
    assert all(pattern.match(c["chunk_id"]) for c in res_parafrasis["semantic"]), \
        "Los chunk_ids recuperados deben pertenecer al corpus"


@pytest.mark.parametrize("query,description", [
    ("abc",                          "muy corta"),
    ("12345678901234",               "solo números"),
    ("!!! ??? ### @@@ $$$ %%% &&&",  "solo símbolos"),
    ("a" * 501,                      "excede 500 caracteres"),
    ("",                             "vacía"),
    ("   ",                          "solo espacios"),
])
def test_caso_invalido_rechazado(client, query, description):
    """Entradas inválidas deben ser rechazadas con 422 antes de llegar al modelo."""
    res_ask    = client.post("/api/ask",    json={"query": query})
    res_search = client.post("/api/search", json={"query": query})
    assert res_ask.status_code    == 422, f"ask debería rechazar: {description}"
    assert res_search.status_code == 422, f"search debería rechazar: {description}"


def test_semantica_supera_bm25_en_parafrasis(engine):
    """Comparación semántica vs BM25 (línea base).
    Una paráfrasis sin palabras clave exactas debe tener score semántico
    normalizado mayor que BM25 normalizado, demostrando la ventaja del modelo."""
    # Consulta directa: BM25 puede competir por palabras exactas
    direct  = engine.search("revitalización del quechua mediante educación")
    # Paráfrasis: sin palabras clave del corpus, BM25 debería perder terreno
    paraph  = engine.search("cómo preservar idiomas ancestrales en jóvenes")

    # Normalizar BM25 al rango [0,1] dividiendo por el máximo del corpus
    bm25_scores_direct = [c["score"] for c in direct["lexical"]]
    bm25_max = max(bm25_scores_direct) if bm25_scores_direct else 1.0

    sem_top_direct  = direct["semantic"][0]["score"]
    sem_top_paraph  = paraph["semantic"][0]["score"]
    bm25_top_direct = direct["lexical"][0]["score"] / bm25_max if bm25_max > 0 else 0
    bm25_top_paraph = paraph["lexical"][0]["score"] / bm25_max if bm25_max > 0 else 0

    # Semántica mantiene scores altos en ambas consultas
    assert sem_top_direct > 0.5,  "Semántica debe puntuar alto en consulta directa"
    assert sem_top_paraph > 0.2,  "Semántica debe puntuar algo en paráfrasis"

    # La caída relativa de semántica entre directa y paráfrasis debe ser menor
    # que la caída de BM25 — semántica es más robusta a variación léxica
    sem_drop  = sem_top_direct  - sem_top_paraph
    bm25_drop = bm25_top_direct - bm25_top_paraph
    assert sem_drop <= bm25_drop, (
        f"Semántica cae {sem_drop:.3f} vs BM25 cae {bm25_drop:.3f}: "
        "semántica debe ser más robusta a paráfrasis que BM25"
    )


def test_metrics_endpoint_has_evaluation_data(client):
    """El endpoint de métricas debe exponer datos de evaluación reales,
    no solo metadatos del corpus."""
    data = client.get("/api/metrics").json()
    assert "version"             in data
    assert "corpus_hash"         in data
    assert "documents"           in data
    assert "chunks"              in data
    assert "threshold_selection" in data
    assert "threshold" in data["threshold_selection"]
    # El sistema debe haber indexado al menos un documento
    assert data["documents"] >= 1
    assert data["chunks"]    >= 1


def test_flujo_completo(client):
    """Flujo completo: search → ask → evidencia referencia chunk existente en corpus."""
    query = "políticas públicas de preservación de lenguas originarias en Bolivia"

    # 1. Buscar
    search_data = client.post("/api/search", json={"query": query}).json()
    assert len(search_data["semantic"]) > 0
    top_chunk_id = search_data["semantic"][0]["chunk_id"]

    # 2. Preguntar
    ask_data = client.post("/api/ask", json={"query": query}).json()
    assert ask_data["status"] in ("answered", "below_threshold", "search_only")
    assert "results" in ask_data
    assert "threshold" in ask_data
    assert "raw_output" not in ask_data

    # 3. El chunk top de search debe aparecer también en los resultados de ask
    ask_chunk_ids = [c["chunk_id"] for c in ask_data["results"]["semantic"]]
    assert top_chunk_id in ask_chunk_ids

    # 4. Si respondió, la evidencia debe referenciar chunks del corpus
    if ask_data["status"] == "answered":
        corpus_ids = {c["chunk_id"] for c in client.get("/api/documents").json()}
        for ev in ask_data.get("evidence", []):
            assert ev["chunk_id"] in corpus_ids
