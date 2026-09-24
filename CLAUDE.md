# CLAUDE.md

AMDRA (Auditable Multimodal Dispute Resolution Agent) is a portfolio and research project. A LangGraph agent investigates **synthetic** credit-card disputes, using transaction lookups, receipt OCR, and version-aware policy retrieval from Chroma. It then asks Claude for a cited recommendation and verifies it. The full architecture, threat model, eval plan and milestones are in `docs/DESIGN.md`; read it before any structural change.

## Commands

```bash
source .venv/bin/activate          # Python >= 3.10; install: pip install -e ".[dev]"
amdra generate                     # rebuild data/synthetic (deterministic, seed 7)
pytest -q                          # offline: in-memory store, hashing embedder, no API key
amdra eval --offline               # rule-based baseline; must stay at 100% on every metric
amdra eval --limit 14              # Claude, one case per scenario (~$0.20)
amdra eval --tag injection         # or --scenario/--tag injection_adversarial/ocr_noise (repeatable)
amdra eval                         # all 66 cases with Claude (~$1)
amdra eval --hybrid / --react / --haiku-routing   # opt-in ablations, see DESIGN.md M2b/M3c/M4d
amdra run D00012 [--review [--checkpoint-db PATH]] [--offline]
pip install -e ".[ui]" && streamlit run src/amdra/ui/app.py   # M5 reviewer UI + eval dashboard
```

Evals that call Claude cost money. Ask before running a full eval or more than about 15 cases, and prefer `--offline`, `--tag` or `--scenario` while iterating. Results are written to `evals/results/<timestamp>.{json,md}`; compare new results against the previous run.

## Layout

- `src/amdra/schemas.py` holds the domain models. `config.py` holds env-driven settings and the token pricing table.
- `data/generate.py` holds the scenario templates and ground-truth labels. `data/policies.py` holds the policy corpus.
- `retrieval/store.py` holds the embedders, the Chroma and in-memory stores, and the metadata filters.
- `tools/authz.py` defines the `@authorized` decorator and the scopes. `tools/toolbox.py` holds the SQLite banking tools, OCR, policy search, and credit.
- `guardrails.py` holds the injection scanner and the untrusted-content spotlighting.
- `llm.py` holds `ClaudeReasoner` (structured output) and `OfflineReasoner` (the baseline and oracle).
- `graph/` holds the state (reducers), the nodes, and `build.py` (the wiring, the `Agent` wrapper, and the checkpointer helpers).
- `evals/runner.py` holds the metrics, sampling and reports. `evals/authz_probes.py` holds the red-team tool probes.
- `ui/` holds the optional Streamlit reviewer UI + eval dashboard (M5, `pip install -e ".[ui]"`).

## Invariants (don't break these without updating DESIGN.md and the tests)

1. **Labels are independent of agent code.** Expected outcomes live only in the scenario templates in `generate.py`; never derive them from `OfflineReasoner` or node logic.
2. **Least privilege.**
   - Every tool is wrapped in `@authorized(scope)` and bound to the dispute's account.
   - Nodes get scopes only through `NODE_SCOPES`.
   - `decide` and `verify` have no scopes.
   - `credit:write` belongs only to `human_review`, which acts only when `human_decision.approved` is set.
3. **The graph calls tools, not the LLM.** The `"react"` investigator (M3c, opt-in via `Settings.investigator`) is the one exception, and even there the model only picks *which* tool to call from a bounded roster the graph hands it — never *whether* to call one, and every tool routes through the same authz wrapper. If you add another model-driven tool-calling path, follow that pattern and add probes to `evals/authz_probes.py`.
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

**All five milestones (M1–M5) in `docs/DESIGN.md` §10 are done.** 33 offline tests pass; `amdra eval --offline` scores 1.000 on every core metric across the full 66-case suite; pricing in `config.PRICING_PER_MTOK` has been verified against live rates. Highlights worth knowing before touching related code:

- **M2b (hybrid retrieval)** found no measurable lift on this corpus — a structural finding (sibling-section ambiguity only case facts can resolve), not an implementation gap. See DESIGN.md §5.
- **M3c (`"react"` investigator)** is a real ablation, not the default — `Settings.investigator` stays `"fixed"` unless explicitly opted in.
- **M4b/M4c (adversarial injection + classifier)** found the regex scanner alone catches 0% of even moderately-obfuscated attempts; the M4c classifier closes most but not all of that gap — the remaining miss traces to OCR never extracting certain text at all when `tesseract` isn't installed, not a detector failure. See DESIGN.md §8.
- **M4d (Haiku routing)** is implemented and verified correct, but the live measurement showed a net cost/latency *loss* on this domain as currently tuned — stays opt-in for exactly that reason. Don't assume it's a win without re-measuring.
- **A pre-existing checkpointer bug** (unregistered-type deserialization warnings, silently swallowed by pytest's default log capture) was found and fixed while building M5 — `graph/build.py`'s `memory_checkpointer()`/`sqlite_checkpointer()` replace bare `MemorySaver()`/`SqliteSaver.from_conn_string()` everywhere. Use those, not the raw constructors, for any new checkpointer usage.

Everything above was measured live, not assumed — re-verify before citing a number from here if meaningfully more code has changed since.

Open design questions (not yet resolved, see DESIGN.md §11): whether `escalate` should count as correct against an expected `deny` when evidence is ambiguous; whether the model should see computed facts, raw records, or both; whether react-mode's `OfflineReasoner` combination needs hardening for every reason code.
