"""
Interactive helper to label relevant_ids in queries.json.
Run AFTER build_index. For each query it shows the top-5 retrieved
chunks so you can decide which ones are truly relevant.

Usage:
    python -m scripts.label_queries
    python -m scripts.label_queries --query-id Q01     # label a single query
    python -m scripts.label_queries --unlabeled        # skip already-labeled queries
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import SearchEngine
from app.settings import DATA_DIR


QUERIES_PATH = DATA_DIR / "queries.json"
TOP_LABEL    = 5   # show more candidates than TOP_K for labelling


def show_chunk(rank: int, chunk: dict, method: str = "sem") -> None:
    print(f"\n  [{rank}][{method}] {chunk['chunk_id']}  score={chunk['score']:.4f}")
    print(f"      {chunk['title']}")
    print(f"      {chunk['heading']}")
    print(f"      {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")


def label_query(engine: SearchEngine, query: dict) -> list[str]:
    print(f"\n{'─'*60}")
    print(f"  {query['query_id']} [{query['split']}]  family={query['family_id']}")
    print(f"  Query: {query['query']}")
    print(f"  Current relevant_ids: {query['relevant_ids']}")

    results = engine.search(query["query"], k=TOP_LABEL)

    print(f"\n  Top {TOP_LABEL} semantic results:")
    for i, chunk in enumerate(results["semantic"], 1):
        show_chunk(i, chunk, "sem")

    print(f"\n  Top {TOP_LABEL} BM25 results:")
    for i, chunk in enumerate(results["lexical"], 1):
        show_chunk(i, chunk, "bm25")

    print(f"\n  Enter the chunk_ids that are relevant (comma-separated).")
    print(f"  Press Enter to keep current: {query['relevant_ids']}")
    print(f"  Type 'skip' to leave unchanged, 'none' to mark as unanswerable.")

    raw = input("  > ").strip()

    if raw.lower() == "skip" or raw == "":
        return query["relevant_ids"]
    if raw.lower() == "none":
        return []

    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Label queries interactively")
    parser.add_argument("--query-id",  type=str, default=None,
                        help="Label only this query_id")
    parser.add_argument("--unlabeled", action="store_true",
                        help="Skip queries that already have relevant_ids")
    args = parser.parse_args()

    if not QUERIES_PATH.exists():
        print(f"queries.json not found at {QUERIES_PATH}")
        sys.exit(1)

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    print("Loading search engine...")
    engine = SearchEngine()
    print(f"Index ready: {len(engine.chunks)} chunks\n")

    changed = 0
    for q in queries:
        if args.query_id and q["query_id"] != args.query_id:
            continue
        if args.unlabeled and q["relevant_ids"]:
            continue

        # Remove internal _note field before processing
        q.pop("_note", None)

        new_ids = label_query(engine, q)
        if new_ids != q["relevant_ids"]:
            q["relevant_ids"] = new_ids
            q["answerable"]   = bool(new_ids)
            changed += 1

    # Save back
    QUERIES_PATH.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'─'*60}")
    print(f"Saved. {changed} query/queries updated.")
    print(f"Run: python -m app.build_index   to recalculate metrics.")


if __name__ == "__main__":
    main()
