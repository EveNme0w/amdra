# CLAUDE.md

AMDRA (Auditable Multimodal Dispute Resolution Agent) is a portfolio and research project. A LangGraph agent investigates **synthetic** credit-card disputes, using transaction lookups, receipt OCR, and version-aware policy retrieval from Chroma. It then asks Claude for a cited recommendation and verifies it. The full architecture, threat model, eval plan and milestones are in `docs/DESIGN.md`; read it before any structural change.

## Commands

```bash
source .venv/bin/activate          # Python >= 3.10; install: pip install -e ".[dev]"
amdra generate                     # rebuild data/synthetic (deterministic, seed 7)
pytest -q                          # offline: in-memory store, hashing embedder, no API key
amdra eval --offline               # rule-based baseline; must stay at 100% on every metric
amdra eval --limit 14              # Claude, one case per scenario (~$0.35)
amdra eval --tag injection         # or --scenario <name> (repeatable)
amdra eval                         # all 42 cases with Claude (~$1)
amdra run D00012 [--review] [--offline]
```

Evals that call Claude cost money. Ask before running a full eval or more than about 15 cases, and prefer `--offline`, `--tag` or `--scenario` while iterating. Results are written to `evals/results/<timestamp>.{json,md}`; compare new results against the previous run.

## Layout

- `src/amdra/schemas.py` holds the domain models. `config.py` holds env-driven settings and the token pricing table.
- `data/generate.py` holds the scenario templates and ground-truth labels. `data/policies.py` holds the policy corpus.
- `retrieval/store.py` holds the embedders, the Chroma and in-memory stores, and the metadata filters.
- `tools/authz.py` defines the `@authorized` decorator and the scopes. `tools/toolbox.py` holds the SQLite banking tools, OCR, policy search, and credit.
- `guardrails.py` holds the injection scanner and the untrusted-content spotlighting.
- `llm.py` holds `ClaudeReasoner` (structured output) and `OfflineReasoner` (the baseline and oracle).
- `graph/` holds the state (reducers), the nodes, and `build.py` (the wiring and the `Agent` wrapper).
- `evals/runner.py` holds the metrics, sampling and reports. `evals/authz_probes.py` holds the red-team tool probes.

## Invariants (don't break these without updating DESIGN.md and the tests)

1. **Labels are independent of agent code.** Expected outcomes live only in the scenario templates in `generate.py`; never derive them from `OfflineReasoner` or node logic.
2. **Least privilege.**
   - Every tool is wrapped in `@authorized(scope)` and bound to the dispute's account.
   - Nodes get scopes only through `NODE_SCOPES`.
   - `decide` and `verify` have no scopes.
   - `credit:write` belongs only to `human_review`, which acts only when `human_decision.approved` is set.
3. **The graph calls tools, not the LLM.** If you add model-driven tool calling (milestone M3), route it through the same authz wrapper and add probes.
4. **Untrusted text** (receipts, narratives) is `Evidence(trusted=False)` and is rendered through `guardrails.render_evidence`. Never splice it into a system prompt.
5. **Citations are verified verbatim** in `verify`. Approvals, escalations, low confidence and injection flags always go to human review.
6. **Observability is append-only.** Nodes return `audit`, `tool_calls` and `llm_usage` through the `@node` wrapper; don't bypass it.
7. **Deterministic data.** Changing the generator changes dispute IDs, so update any hard-coded IDs in docs and examples.
8. **Tests stay offline.** They need no network and no API key.

## Conventions

- Python 3.10+, type hints, and pydantic v2 models. Keep the line length at 100 or less (ruff).
- Money is stored as integer cents. Dates in metadata are YYYYMMDD integers so Chroma can filter them.
- Every new scenario needs a template in `SCENARIOS`, the expected outcome and policy section, tags where relevant, and a passing offline baseline.
- Every new metric goes in `score_case` or `summarize` and in the docstring at the top of `evals/runner.py`.
- Never commit `.env` or anything under `data/synthetic/`, `.chroma/` or `evals/results/`, all of which are gitignored.

## Current state (Sept 2026)

- **M1 skeleton is complete.** 18 offline tests pass.
- **First Claude run** (`claude-sonnet-4-5`, 10 fraud and duplicate cases) was 100% accurate with 100% citation validity. p50 latency was about 12 s, and cost about $0.023 per case, with about 870 output tokens per case.
- **Pricing check needed:** the rates in `config.PRICING_PER_MTOK` are assumptions and need to be verified.

Suggested next steps:

1. Run the full Claude eval and record a baseline.
2. Cut latency and cost with a shorter rationale, token caps, or Haiku routing.
3. M2: noisy receipts (blur, rotation), OCR confidence, and a vision fallback.
4. Harder and adversarial cases: conflicting evidence, and injections that dodge the regex scanner.
5. An ablation comparing raw records with computed facts in the prompt.
