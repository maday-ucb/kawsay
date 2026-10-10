"""Read the Markdown corpus and expose traceable chunks for indexing."""
import hashlib
import json
import re
from pathlib import Path
from app.settings import DOCUMENT_DIR, DATA_DIR


def read_chunks(directory: Path = DOCUMENT_DIR) -> list[dict]:
    """
    Parse every D##_*.md file into a list of section chunks.
    Each chunk carries enough metadata to cite its source.
    """
    chunks = []
    for path in sorted(directory.glob("D[0-9]*_*.md")):
        content = path.read_text(encoding="utf-8")
        lines   = content.splitlines()

        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"Missing document title in: {path.name}")

        title       = lines[0][2:].strip()
        document_id = path.stem.split("_")[0]

        if not re.fullmatch(r"D[0-9]{3}", document_id):
            raise ValueError(f"Filename must start with D + three digits: {path.name}")

        # Extract doc_type from the HTML comment injected by ingest_pdfs
        doc_type = "otro"
        comment_match = re.search(r"<!--\s*type:(\w+)", content)
        if comment_match:
            doc_type = comment_match.group(1)

        # Split on ## headings
        sections = re.split(r"^## ", content, flags=re.MULTILINE)[1:]
        if not sections:
            raise ValueError(f"No sections (##) found in: {path.name}")

        for number, section in enumerate(sections, 1):
            heading, _, body = section.partition("\n")
            # Collapse whitespace but keep the text as a single paragraph
            body = " ".join(body.split())
            if not body:
                continue
            chunks.append({
                "chunk_id":   f"{document_id}-S{number}",
                "document_id": document_id,
                "filename":   path.name,
                "title":      title,
                "doc_type":   doc_type,
                "heading":    heading.strip(),
                "text":       body,
            })

    if not chunks:
        raise ValueError(
            "No documents found in data/documents/. "
            "Run scripts/ingest_pdfs.py first."
        )

    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs detected. Check document filenames.")

    return chunks


def corpus_hash(chunks: list[dict]) -> str:
    """Stable fingerprint of the corpus content."""
    return hashlib.sha256(
        json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_queries(chunks: list[dict]) -> list[dict]:
    """Load and validate queries.json against the current corpus."""
    queries_path = DATA_DIR / "queries.json"
    if not queries_path.exists():
        return []   # queries are optional until evaluation

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    ids = {c["chunk_id"] for c in chunks}

    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Query IDs must be unique in queries.json.")

    for q in queries:
        if q.get("split") not in ("validation", "test"):
            raise ValueError(f"Query {q['query_id']} has invalid split.")
        if not q.get("query", "").strip():
            raise ValueError(f"Query {q['query_id']} has empty text.")
        if not set(q.get("relevant_ids", [])) <= ids:
            unknown = set(q.get("relevant_ids", [])) - ids
            raise ValueError(
                f"Query {q['query_id']} references unknown chunk IDs: {unknown}"
            )
        if q.get("answerable") != bool(q.get("relevant_ids")):
            raise ValueError(
                f"Query {q['query_id']}: 'answerable' must match whether "
                f"'relevant_ids' is non-empty."
            )

    # Ensure no family crosses validation and test
    val_families  = {q["family_id"] for q in queries if q["split"] == "validation"}
    test_families = {q["family_id"] for q in queries if q["split"] == "test"}
    overlap = val_families & test_families
    if overlap:
        raise ValueError(f"Family IDs appear in both splits: {overlap}")

    return queries
