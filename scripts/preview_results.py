"""
Muestra los top-10 resultados (sem + bm25) para Q01-Q18 de una sola vez.
Úsalo para decidir qué chunk_ids poner en queries.json antes de re-etiquetar.

Uso:
    python -m scripts.preview_results
    python -m scripts.preview_results --query-id Q01
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import SearchEngine
from app.settings import DATA_DIR

QUERIES_PATH = DATA_DIR / "queries.json"
TOP = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", default=None)
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    print("Cargando índice...")
    engine = SearchEngine()
    print(f"Listo: {len(engine.chunks)} chunks\n")

    for q in queries:
        if args.query_id and q["query_id"] != args.query_id:
            continue
        # Solo Q01-Q18 (answerable pendiente)
        if not args.query_id and q["query_id"] in {f"Q{i:02d}" for i in range(19, 25)}:
            continue

        print(f"\n{'='*70}")
        print(f"  {q['query_id']} [{q['split']}]  family={q['family_id']}")
        print(f"  {q['query']}")
        print(f"  relevant_ids actuales: {q['relevant_ids']}")

        results = engine.search(q["query"], k=TOP)

        print(f"\n  -- Semantico --")
        for i, c in enumerate(results["semantic"], 1):
            print(f"  [{i}] {c['chunk_id']}  score={c['score']:.4f}")
            print(f"       {c['title'][:80]}")
            print(f"       {c['text'][:150].strip()}...")

        print(f"\n  -- BM25 --")
        for i, c in enumerate(results["lexical"], 1):
            print(f"  [{i}] {c['chunk_id']}  score={c['score']:.4f}")
            print(f"       {c['title'][:80]}")
            print(f"       {c['text'][:150].strip()}...")


if __name__ == "__main__":
    main()
