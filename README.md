# AMDRA: Auditable Multimodal Dispute Resolution Agent

A LangGraph agent that investigates **synthetic** credit-card disputes. It gathers transactions, reads receipt images with OCR, retrieves version-aware bank policy from Chroma, and asks Claude for a cited recommendation. It then verifies every citation and sends payouts to a human for approval.

See [docs/DESIGN.md](docs/DESIGN.md) for architecture, security model, eval plan and milestones.

## Quick start

```bash
brew install tesseract              # optional; without it OCR falls back to sidecar text
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env                # add ANTHROPIC_API_KEY

amdra generate                      # data/synthetic: bank.db, receipts/*.png, policies/*.md, cases.jsonl
amdra eval --offline                # rule-based baseline, no API key (should be 100%)
amdra eval --limit 14               # Claude on one case per scenario (~$0.20)
amdra eval --tag injection          # or --tag injection_adversarial / ocr_noise
amdra eval --hybrid / --react / --haiku-routing   # opt-in ablations, see docs/DESIGN.md
amdra run D00012 --review           # one case, pausing at the human-review gate
pytest

uv pip install -e ".[ui]"           # optional: reviewer UI + eval dashboard
streamlit run src/amdra/ui/app.py
```

## Layout

```
src/amdra/
  schemas.py            domain models (Dispute, Evidence, Recommendation, ...)
  config.py             env-driven settings, token pricing
  data/generate.py      seeded scenario generator + receipt image rendering
  data/policies.py      synthetic policy corpus (versioned, with a debit distractor)
  retrieval/store.py    embedders, Chroma / in-memory stores, metadata filters
  tools/authz.py        per-node scopes + account binding for every tool call
  tools/toolbox.py      SQLite banking tools, OCR, policy search, provisional credit
  guardrails.py         injection scanner + untrusted-content spotlighting
  llm.py                ClaudeReasoner (structured output) and OfflineReasoner baseline
  graph/                LangGraph state, nodes, wiring (verify/retry, HITL interrupt), checkpointers
  evals/                eval runner + tool-authorization red-team probes
  ui/                   optional Streamlit reviewer UI + eval dashboard (pip install -e ".[ui]")
tests/                  pytest suite (offline, in-memory, no API key)
```

## Status

All five milestones in [docs/DESIGN.md](docs/DESIGN.md) §10 are complete: the deterministic graph with an offline rule-based baseline (M1); hybrid retrieval, noisy-receipt OCR confidence, and a vision-model fallback (M2); LangSmith tracing, a persistent checkpointer, and an opt-in ReAct investigator ablation (M3); an adversarial injection suite, an LLM classifier second opinion, and cost/latency budgets with opt-in Haiku routing (M4); and a Streamlit reviewer UI + eval dashboard (M5). 33 offline tests pass; `amdra eval --offline` scores 100% on every core metric across the full 66-case suite. See `docs/DESIGN.md` for the live-measured findings behind each milestone (several surprising, documented rather than smoothed over) and the open questions still on the table.
