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
amdra eval --limit 14               # Claude on one case per scenario (~$0.35)
amdra eval --tag injection          # just the prompt-injection cases
amdra run D00012 --review           # one case, pausing at the human-review gate
pytest
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
  graph/                LangGraph state, nodes, and wiring (verify/retry, HITL interrupt)
  evals/                eval runner + tool-authorization red-team probes
tests/                  pytest suite (offline, in-memory, no API key)
```

## Status

v0.1 skeleton. The core logic and test suite were verified offline with a minimal LangGraph stand-in, because PyPI was unreachable in the build environment. Run `pytest` after installing to confirm against the real `langgraph` and `chromadb` packages.
