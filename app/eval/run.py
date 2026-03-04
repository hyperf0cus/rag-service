"""
RAG evaluation CLI.

Usage
-----
    python -m app.eval.run \\
        --tenant acme \\
        --dataset data/eval/golden.jsonl \\
        --k 3 5 10

golden.jsonl format (one JSON object per line)::

    {
      "query": "How do I reset my password?",
      "relevant_chunk_ids": ["doc_001_0002", "doc_001_0003"],  // preferred
      "relevant_doc_ids":   ["doc_001"]                        // fallback
    }

At least one of ``relevant_chunk_ids`` or ``relevant_doc_ids`` must be
non-empty.  When both are present ``relevant_chunk_ids`` takes precedence.

Output
------
- Console: rich table with per-query and aggregate metrics.
- File: JSON report written to ``<output_dir>/eval_<tenant>_<timestamp>.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Make the repo root importable when run as ``python -m app.eval.run``
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.core.logging import setup_logging  # noqa: E402
from app.eval.metrics import (  # noqa: E402
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from app.rag.retrieval import retrieve  # noqa: E402


# ── CLI arg parsing ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.run",
        description="Evaluate RAG retrieval quality against a golden dataset.",
    )
    parser.add_argument("--tenant", required=True, help="Tenant ID to evaluate.")
    parser.add_argument("--dataset", required=True, help="Path to golden.jsonl.")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        metavar="K",
        help="k values for Recall@k (default: 3 5 10).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write the JSON report (default: reports/).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Similarity threshold (0.0 disables filtering; default: 0.0).",
    )
    return parser.parse_args()


# ── Core evaluation loop ───────────────────────────────────────────────────────

async def run_eval(
    tenant_id: str,
    dataset_path: str,
    k_values: list[int],
    output_dir: str,
    score_threshold: float = 0.0,
) -> None:
    setup_logging("WARNING")  # suppress INFO noise during eval

    if not os.path.isfile(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as fh:
        examples = [json.loads(line) for line in fh if line.strip()]

    if not examples:
        print("[ERROR] Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    max_k = max(k_values)
    print(f"\nEvaluating {len(examples)} queries | tenant={tenant_id} | k={k_values}\n")

    per_query: list[dict] = []
    rr_pairs: list[tuple[set[str], list[str]]] = []
    recall_sums: dict[int, float] = {k: 0.0 for k in k_values}

    for idx, example in enumerate(examples, 1):
        query: str = example.get("query", "")
        if not query:
            print(f"  [{idx}/{len(examples)}] SKIP – missing 'query' field")
            continue

        # Ground-truth IDs: prefer chunk-level, fall back to doc-level.
        relevant_chunk_ids = set(example.get("relevant_chunk_ids") or [])
        relevant_doc_ids = set(example.get("relevant_doc_ids") or [])
        if not relevant_chunk_ids and not relevant_doc_ids:
            print(f"  [{idx}/{len(examples)}] SKIP – no relevant_chunk_ids or relevant_doc_ids")
            continue

        try:
            chunks = await retrieve(
                query=query,
                tenant_id=tenant_id,
                top_k=max_k,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            print(f"  [{idx}/{len(examples)}] ERROR – {exc}")
            continue

        # Choose ID level to evaluate on
        if relevant_chunk_ids:
            relevant = relevant_chunk_ids
            retrieved = [c.chunk_id for c in chunks]
        else:
            relevant = relevant_doc_ids
            retrieved = [c.doc_id for c in chunks]

        recall_vals = {k: recall_at_k(relevant, retrieved, k) for k in k_values}
        rr = reciprocal_rank(relevant, retrieved)

        for k, v in recall_vals.items():
            recall_sums[k] += v
        rr_pairs.append((relevant, retrieved))

        per_query.append(
            {
                "query": query,
                "relevant_ids": sorted(relevant),
                "retrieved_ids": retrieved,
                "recall_at_k": recall_vals,
                "reciprocal_rank": round(rr, 4),
                "top_score": round(chunks[0].score, 4) if chunks else 0.0,
                "n_retrieved": len(chunks),
            }
        )

        r_str = "  ".join(f"R@{k}={v:.3f}" for k, v in recall_vals.items())
        print(f"  [{idx}/{len(examples)}] {r_str}  RR={rr:.3f}  | {query[:70]}")

    n = len(per_query)
    if n == 0:
        print("\nNo queries evaluated successfully.")
        return

    avg_recall = {k: round(recall_sums[k] / n, 4) for k in k_values}
    mrr = round(mean_reciprocal_rank(rr_pairs), 4)
    hit_rates = {k: round(hit_rate_at_k(rr_pairs, k), 4) for k in k_values}

    # ── Console summary ────────────────────────────────────────────────────────
    sep = "─" * 44
    print(f"\n{sep}")
    print(f"  {'METRIC':<22} {'VALUE':>8}")
    print(sep)
    for k in k_values:
        print(f"  {'Recall@'+str(k):<22} {avg_recall[k]:>8.4f}")
    for k in k_values:
        print(f"  {'HitRate@'+str(k):<22} {hit_rates[k]:>8.4f}")
    print(f"  {'MRR':<22} {mrr:>8.4f}")
    print(f"  {'Queries evaluated':<22} {n:>8}")
    print(sep)

    # ── JSON report ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tenant_id": tenant_id,
        "dataset": dataset_path,
        "n_queries": n,
        "k_values": k_values,
        "avg_recall_at_k": avg_recall,
        "avg_hit_rate_at_k": hit_rates,
        "mrr": mrr,
        "per_query": per_query,
    }
    ts = int(time.time())
    report_path = os.path.join(output_dir, f"eval_{tenant_id}_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  Report saved → {report_path}\n")


# ── Entry-point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    asyncio.run(
        run_eval(
            tenant_id=args.tenant,
            dataset_path=args.dataset,
            k_values=sorted(set(args.k)),
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
        )
    )


if __name__ == "__main__":
    main()
