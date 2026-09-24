"""Command line entry point.

    amdra generate                 # build synthetic data
    amdra run D00012               # investigate one dispute and print the report
    amdra run D00012 --review      # pause at the human-review gate and ask for a decision
    amdra run D00012 --review --checkpoint-db out.sqlite  # persist the pause across a restart
    amdra eval --limit 14          # one case per scenario (see --scenario, --tag)
    amdra eval --limit 14 --react  # same cases, via the M3c ReAct investigator instead
Add --offline to any command to use the rule-based reasoner and hashing embedder.
"""
from __future__ import annotations

import argparse
import json

from amdra.config import Settings


def _settings(offline: bool, hybrid: bool = False, react: bool = False) -> Settings:
    s = Settings()
    if offline:
        s.llm, s.embedder = "offline", "hashing"
    if hybrid:
        s.hybrid_retrieval = True
    if react:
        s.investigator = "react"
    return s


def report(state: dict) -> dict:
    rec = state.get("recommendation")
    return {
        "dispute_id": state["dispute"].dispute_id,
        "status": state.get("status"),
        "recommendation": rec.model_dump(mode="json") if rec else None,
        "verification": state.get("verification"),
        "needs_human_review": state.get("needs_human_review"),
        "review_reasons": state.get("review_reasons"),
        "injection_flags": state.get("injection_flags"),
        "actions": state.get("actions"),
        "audit": state.get("audit"),
        "tool_calls": [{k: c[k] for k in ("node", "tool", "status")} for c in state.get("tool_calls", [])],
        "llm_usage": state.get("llm_usage"),
    }


def cmd_run(args) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from amdra.data.generate import load_cases

    s = _settings(args.offline)
    case = next((c for c in load_cases(s.cases_path) if c.dispute.dispute_id == args.dispute_id), None)
    if case is None:
        raise SystemExit(f"unknown dispute {args.dispute_id}")

    if args.checkpoint_db:
        # Persistent (M3b): survives a process restart mid human-review pause. Not the default —
        # tests and eval runs keep MemorySaver so they stay fast and side-effect-free.
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(args.checkpoint_db) as checkpointer:
            _run_case(s, case, checkpointer, args)
    else:
        _run_case(s, case, MemorySaver(), args)


def _run_case(s: Settings, case, checkpointer, args) -> None:
    from amdra.graph.build import Agent

    agent = Agent.create(s, checkpointer=checkpointer, interrupt_for_review=args.review)
    config = {"configurable": {"thread_id": case.dispute.dispute_id}}
    state = agent.graph.invoke({"dispute": case.dispute}, config=config)

    if args.review and agent.graph.get_state(config).next:
        rec = state.get("recommendation")
        print(f"\nPaused for human review: {state.get('review_reasons')}")
        if rec:
            print(f"Recommendation: {rec.outcome.value} under {rec.policy_section} "
                  f"(confidence {rec.confidence:.2f})\n{rec.rationale}")
        answer = input("Approve this recommendation? [y/N] ").strip().lower() == "y"
        agent.graph.update_state(config, {"human_decision": {
            "approved": answer, "reviewer": "cli-user", "note": ""}})
        state = agent.graph.invoke(None, config=config)

    print(json.dumps(report(state), indent=2, default=str))
    print(f"\nexpected: {case.expected_outcome.value} under {case.expected_policy_section}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="amdra")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--per-scenario", type=int, default=3)
    g.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run")
    r.add_argument("dispute_id")
    r.add_argument("--review", action="store_true")
    r.add_argument("--offline", action="store_true")
    r.add_argument("--checkpoint-db", help="SQLite file for a persistent checkpointer "
                   "(default: in-memory, lost when the process exits)")
    e = sub.add_parser("eval")
    from amdra.evals import add_eval_args

    add_eval_args(e)
    args = ap.parse_args()

    if args.cmd == "generate":
        from amdra.data.generate import generate

        print(generate(Settings().data_dir, args.per_scenario, args.seed))
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "eval":
        s = _settings(args.offline, args.hybrid, args.react)
        if args.retrieval_only:
            from amdra.data.generate import load_cases
            from amdra.evals.runner import evaluate_retrieval, select_cases

            cases = select_cases(load_cases(s.cases_path), args.limit, args.scenario, args.tag)
            print(json.dumps(evaluate_retrieval(s, cases), indent=2))
        else:
            from amdra.evals.runner import run_eval

            run_eval(s, args.limit, scenarios=args.scenario, tags=args.tag)


if __name__ == "__main__":
    main()
