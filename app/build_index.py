"""
Build the search index from the Markdown corpus.
Run this every time documents change.

Usage:
    python -m app.build_index
"""
import hashlib
import json
import joblib
import numpy as np
from pathlib import Path

from app.documents import read_chunks, corpus_hash, read_queries
from app.retrieval import load_encoder, encode_passages, encode_query, ranked_indices
from app.evaluation import ranking_metrics, select_gate, gate_metrics
from app.settings import ARTIFACT_DIR, DATA_DIR, TOP_K, DEFAULT_THRESHOLD


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_bm25(chunks: list[dict]):
    """Build a BM25 index from chunk texts."""
    from rank_bm25 import BM25Okapi
    tokenized = [
        (c["title"] + " " + c["heading"] + " " + c["text"]).lower().split()
        for c in chunks
    ]
    return BM25Okapi(tokenized)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip re-encoding; reuse existing index.joblib")
    args = parser.parse_args()

    index_path = ARTIFACT_DIR / "index.joblib"

    if args.eval_only:
        if not index_path.exists():
            print("ERROR: index.joblib not found. Run without --eval-only first.")
            return
        print("Loading existing index (--eval-only)...")
        index = joblib.load(index_path)
        chunks  = index["chunks"]
        queries = read_queries(chunks)
        encoder = load_encoder()
        digest  = index["corpus_hash"]
        print(f"  {len(chunks)} chunks, {len(queries)} queries")
    else:
        print("Loading corpus...")
        chunks  = read_chunks()
        queries = read_queries(chunks)
        print(f"  {len(chunks)} chunks from "
              f"{len({c['document_id'] for c in chunks})} documents")
        if queries:
            print(f"  {len(queries)} queries loaded")
        else:
            print("  No queries.json found — skipping evaluation metrics")

        print("\nLoading embedding model...")
        encoder = load_encoder()

        texts = [
            f"{c['title']}. {c['heading']}. {c['text']}"
            for c in chunks
        ]

        print("Encoding corpus (this may take a few minutes)...")
        embeddings = encode_passages(encoder, texts, batch_size=32)

        print("Building BM25 index...")
        bm25 = build_bm25(chunks)

        digest = corpus_hash(chunks)
        index  = {
            "chunks":      chunks,
            "embeddings":  embeddings,
            "bm25":        bm25,
            "corpus_hash": digest,
            "version":     "kawsay-v1-" + digest[:8],
        }

        ARTIFACT_DIR.mkdir(exist_ok=True)
        joblib.dump(index, ARTIFACT_DIR / "index.joblib", compress=3)
        print(f"\nIndex saved → artifacts/index.joblib")

    # ── Evaluation (only if queries.json exists) ──────────────────────────────
    if not queries:
        report = {
            "version":      index["version"],
            "corpus_hash":  digest,
            "documents":    len({c["document_id"] for c in chunks}),
            "chunks":       len(chunks),
            "query_count":  0,
            "partitions":   {},
            "threshold_selection": {"threshold": DEFAULT_THRESHOLD},
        }
        save_json(ARTIFACT_DIR / "metrics.json", report)
        print("Metrics saved (no queries — default threshold used)")
        return

    print("\nRunning retrieval evaluation...")

    from app.retrieval import SearchEngine
    engine = SearchEngine(encoder)
    rows   = []
    for q in queries:
        results    = engine.search(q["query"])
        top_score  = results["semantic"][0]["score"]
        rows.append({
            **q,
            "top_score": top_score,
            "rankings": {
                method: [c["chunk_id"] for c in ranking]
                for method, ranking in results.items()
            },
        })

    # Select gate threshold on validation set
    validation = [r for r in rows if r["split"] == "validation"]
    threshold  = select_gate(
        [r["answerable"] for r in validation],
        [r["top_score"]  for r in validation],
    ) if validation else {"threshold": DEFAULT_THRESHOLD}

    # Build report
    queries_sha = hashlib.sha256(
        (DATA_DIR / "queries.json").read_bytes()
    ).hexdigest()

    report = {
        "version":            index["version"],
        "corpus_hash":        digest,
        "queries_sha256":     queries_sha,
        "documents":          len({c["document_id"] for c in chunks}),
        "chunks":             len(chunks),
        "query_count":        len(queries),
        "threshold_selection": threshold,
        "partitions":         {},
    }

    for split in ("validation", "test"):
        subset    = [r for r in rows if r["split"] == split]
        answerable = [r for r in subset if r["answerable"]]
        partition  = {
            "queries":    len(subset),
            "answerable": len(answerable),
            "retrieval":  {},
        }

        for method in ("semantic", "lexical"):
            if answerable:
                metrics_list = [
                    ranking_metrics(r["rankings"][method], r["relevant_ids"])
                    for r in answerable
                ]
                partition["retrieval"][method] = {
                    key: float(np.mean([m[key] for m in metrics_list]))
                    for key in metrics_list[0]
                }

        partition["gate"] = gate_metrics(
            [r["answerable"] for r in subset],
            [r["top_score"]  for r in subset],
            threshold["threshold"],
        )
        report["partitions"][split] = partition

    save_json(ARTIFACT_DIR / "metrics.json", report)
    save_json(ARTIFACT_DIR / "retrieval_predictions.json", rows)

    # ── Human-readable evaluation report ─────────────────────────────────────
    lines = [
        "# Evaluación de recuperación — Kawsay", "",
        "Corpus de revitalización lingüística. "
        "Los modelos de embeddings no se ajustaron con estas consultas.", "",
    ]

    for split in ("validation", "test"):
        p = report["partitions"].get(split, {})
        if not p:
            continue
        lines += [f"## {split.capitalize()}", ""]
        lines += [
            "| Métrica | Semántica | BM25 |",
            "|---|---:|---:|",
        ]
        sem = p["retrieval"].get("semantic", {})
        lex = p["retrieval"].get("lexical", {})
        for metric in ("ndcg_at_5", "recall_at_5", "precision_at_5", "mrr_at_5"):
            lines.append(
                f"| {metric} "
                f"| {sem.get(metric, 0):.4f} "
                f"| {lex.get(metric, 0):.4f} |"
            )
        lines.append("")

    lines += [
        f"Umbral de paso a generación (validación): "
        f"{threshold['threshold']:.6f}.", "",
        "Las métricas de ranking se calculan solo sobre consultas respondibles.",
    ]
    (ARTIFACT_DIR / "evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    test_p = report["partitions"].get("test", {})
    if test_p:
        sem = test_p["retrieval"].get("semantic", {})
        lex = test_p["retrieval"].get("lexical", {})
        print(f"\n── Test results ─────────────────────────")
        print(f"  {'Metric':<20} {'Semantic':>10} {'BM25':>10}")
        print(f"  {'-'*42}")
        for metric in ("ndcg_at_5", "recall_at_5", "precision_at_5", "mrr_at_5"):
            print(f"  {metric:<20} "
                  f"{sem.get(metric, 0):>10.4f} "
                  f"{lex.get(metric, 0):>10.4f}")
        print(f"\n  Gate threshold : {threshold['threshold']:.4f}")
        print(f"  Gate F1        : {test_p['gate'].get('f1', 0):.4f}")

    print(f"\nDone. Version: {index['version']}")


if __name__ == "__main__":
    main()
