"""Evaluation suite. Kept import-light so the CLI can build its parser cheaply."""
import argparse


def add_eval_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--limit", type=int, default=None,
                    help="max cases, sampled round-robin across scenarios (tagged first)")
    ap.add_argument("--scenario", action="append", help="only this scenario (repeatable)")
    ap.add_argument("--tag", action="append", help="only cases with this tag, e.g. injection")
    ap.add_argument("--offline", action="store_true", help="rule-based reasoner, hashing embedder")
    ap.add_argument("--hybrid", action="store_true",
                    help="hybrid BM25 + dense retrieval (reciprocal rank fusion)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip the graph/LLM; report recall@1, recall@k and MRR for the "
                         "reason-code policy search alone")
    ap.add_argument("--react", action="store_true",
                    help="M3c: use the ReAct investigator subgraph instead of the fixed "
                         "gather_transactions/gather_documents pipeline (ablation, needs an API key)")
