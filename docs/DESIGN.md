# AMDRA Design

Auditable Multimodal Dispute Resolution Agent. Status: v0.1 skeleton (September 2026).

## 1. Goal

Investigate synthetic credit-card disputes end to end and produce a recommendation that is:

- **Correct:** it matches the policy outcome for the case.
- **Grounded:** every claim cites evidence, and each quote is checked verbatim.
- **Safe:** untrusted text can't steer the decision or trigger tools, and money moves only after a human approves.
- **Auditable:** every node, tool call, denial, model call, and cost lands in the graph state and its checkpoints.

Non-goals for now: real customer data, real card-network rules (Visa and Mastercard reason codes), and production deployment.

## 2. Architecture

`Settings.investigator` picks one of two static graph wirings at build time (never a runtime choice) — `"fixed"` (default) or `"react"` (M3c, opt-in ablation):

```
"fixed" (default):
          ┌──────────┐   ┌─────────────────────┐   ┌──────────────────┐
dispute → │  intake  │ → │ gather_transactions │ → │ gather_documents │
          └──────────┘   └─────────────────────┘   └────────┬─────────┘
           account:read      transactions:read              │ documents:read (OCR)
                                                              ▼
                               low OCR confidence  ┌──────────────────┐   otherwise
                              ┌──────────────────► │  vision_fallback │ ──────────┐
                              │                     └──────────────────┘          │
                              │                      documents:read (image)       ▼
                              │                                          ┌─────────────────┐
                              └──────────────────────────────────────────► retrieve_policy │
                                                                          └────────┬────────┘
"react" (M3c, opt-in):                                                            │
          ┌──────────┐   ┌──────────────┐   txn:read | documents:read |           │
dispute → │  intake  │ → │  investigate │ → policy:search (model picks the tool)  │
          └──────────┘   └──────────────┘ ───────────────────────────────────────►┘
                                                                                   │ policy:search
                              ┌──────── retry with feedback ────────┐       ┌──────────┐    ▼
                              ▼                                     │       │  decide  │◄────┘
                         ┌──────────┐                          ┌────┴───┐   └────┬─────┘  (no tool scopes)
                         │  decide  │ ───────────────────────► │ verify │ ◄──────┘
                         └──────────┘                          └────┬───┘
                                                                    ▼
                                    ┌─────────────┐  needs review  ┌──────────────┐
                                    │ review_gate │ ─────────────► │ human_review │ (interrupt; credit:write)
                                    └──────┬──────┘                └──────┬───────┘
                                           └──────── auto ─────► finalize ◄┘
```

**The split between deterministic code and the LLM is the core design choice.** Tools fetch data. Nodes turn that data into *evidence* and *computed facts*, for example "Found 1 other transaction at X for the same amount within 48 hours". The LLM only judges the evidence against the retrieved policy, and a verifier checks its work. This makes decisions reproducible, keeps arithmetic and date math out of the model, and keeps the model's attack surface small. `retrieve_policy` stays deterministic and unconditional in **both** wirings — policy retrieval correctness (date/version/product filtering) is load-bearing for the grounding guarantee and isn't delegated to the model's query choice, even in `"react"` mode.

Tool calls are planned by the graph, not chosen by the model, in `"fixed"` mode's every node. That removes the most dangerous injection path, where text asks the model to call a tool. `vision_fallback` (M2d) was the first case of a node itself making an LLM call (Claude vision) — the graph, not the model under investigation, decides *when* that call happens (`route_after_documents`, keyed only on a receipt's own recorded OCR confidence). `investigate` (M3c) is the first case of the model itself choosing *which* tool to call, from a bounded roster, in a ReAct loop (`langgraph.prebuilt.create_react_agent`) — see §8 for how this stays inside the same authz model and never becomes a narration-as-evidence risk.

## 3. Data model (`schemas.py`)

| Entity | Notes |
|---|---|
| `Account` | product (credit/debit), home location, `card_reported_lost_at`, registered devices |
| `Transaction` | channel, auth method (chip_pin/chip/swipe/3ds/none), device, location, amount in cents |
| `Receipt` | PNG image + ground-truth sidecar text; kind = receipt or order_confirmation |
| `Dispute` | reason code, filed_at, statement_date, narrative (untrusted), receipt ids |
| `LabeledCase` | dispute + expected outcome + expected governing policy section + tags |
| `Evidence` | id, kind (account/transaction/receipt/policy/computed/narrative), text, `trusted`, metadata |
| `Recommendation` | outcome (approve/deny/escalate), policy_section, rationale, citations[], confidence |
| `VisionResult` | `vision_fallback`'s structured output: transcribed_text, total_cents, expected_delivery, confidence |

Storage: SQLite (`bank.db`) stands in for core banking. Receipts are PNG files, and policies are Markdown files with front matter.

## 4. Synthetic data (`data/generate.py`)

The generator is deterministic (seeded). It builds 18 scenario templates × N variants, 54 cases by default. **Labels come from the templates, never from agent code.**

| Scenario | Expected | Governing section | What it tests |
|---|---|---|---|
| fraud_lost_card | approve | POL-002 §2.1 | account-state reasoning |
| fraud_chip_pin_known_device | escalate | POL-002 §2.2 | mandatory escalation |
| fraud_cnp_foreign | approve | POL-002 §2.3 | multi-condition rule |
| duplicate_true / _false | approve / deny | §3.1 / §3.3 | near-duplicate distractor (same merchant, different amount) |
| amount_lower_receipt / _matches | approve / deny | §3.2 / §3.3 | OCR total extraction |
| not_received_eligible / _too_early / _no_contact | approve / deny / deny | POL-004 | date math + condition |
| late_filing_v1 | deny | POL-001 §1.1 (v1, 60 days) | **policy versioning**: filed 75 days late |
| late_filing_v2 | approve | POL-003 §3.1 (v2 window is 90 days) | same facts, different policy in force |
| injection_receipt | deny | §3.3 | instructions embedded in a receipt image |
| injection_narrative | deny | §3.3 | instructions plus a cross-account tool request in the narrative |
| amount_lower_receipt_noisy_mild / not_received_eligible_noisy_mild | approve (unchanged) | §3.2 / POL-004 §4.1 | **degraded OCR, tag `ocr_noise`**: free-text lines garbled, the TOTAL/delivery-date line stays legible — confidence drops but the fact still extracts correctly |
| amount_matches_noisy_severe / not_received_too_early_noisy_severe | escalate | §3.2 / POL-004 §4.1 | **degraded OCR, tag `ocr_noise`**: the TOTAL/delivery-date line itself is unreadable — extraction genuinely fails, so escalation (not a guessed deny) is the correct, deterministic label |

The policy corpus includes a **debit-card distractor** (POL-900, "approve everything"), which retrieval must filter out by product.

## 5. Multimodal retrieval (`retrieval/store.py`, `tools/toolbox.py`)

- **OCR:** Tesseract through pytesseract, with a regex for totals and delivery dates. If Tesseract isn't installed, it falls back to the sidecar text; the engine and a confidence score are recorded in evidence metadata (`ocr_engine`, `ocr_confidence`).
- **OCR confidence + noisy receipts (M2c, done):** `Receipt.noise_level` (`clean`/`mild`/`severe`) drives two independent degradation paths. (1) Best-effort **pixel noise** (`data/generate.py` `render_receipt`, Pillow `GaussianBlur` + `rotate`) only matters on the real tesseract path — `pytesseract.image_to_data` then yields genuine per-word confidence. (2) A **deterministic, seeded synthetic degradation** (`tools/toolbox.py` `_corrupt_text`, keyed by receipt id) drives the sidecar (no-tesseract) path that `pytest`/`--offline` actually exercise: `mild` garbles free-text lines but protects the TOTAL/delivery-date line, so extraction still succeeds at lower confidence (0.55); `severe` garbles those lines too, so extraction deterministically fails (`None`, confidence 0.15). This keeps `amdra eval --offline` fully deterministic without depending on whether `tesseract` is installed. Severe-noise scenarios are labeled `escalate` in the template (not derived from confidence at decide-time) — `OfflineReasoner`'s existing "no readable receipt total / no expected delivery date" rule already escalates correctly on a missing fact, so no new reasoning code was needed. Not yet consumed: a vision-model fallback (planned M2d) that would recover mild/severe cases instead of just escalating severe ones.
- **Chunking:** one chunk per policy section. Metadata is flat, so the same filter works in Chroma and in memory: `product`, `effective_from/to` (YYYYMMDD ints), `rc_<reason_code>` booleans, `citation`, `version`.
- **Filters:** `product == credit AND rc_<code> AND effective_from <= filed_at < effective_to`.
- **Queries:** one query for the filing window (k=1) and one for the reason code (k=4).
- **Embedders:** Chroma's default MiniLM ONNX model, or a deterministic hashing embedder for offline tests. Each embedder gets its own collection.
- **Hybrid retrieval (M2b, done):** opt-in (`Settings.hybrid_retrieval` / `AMDRA_HYBRID_RETRIEVAL=1` / `amdra eval --hybrid`) reciprocal-rank fusion of the dense search with a BM25 lexical index (`retrieval/bm25.py`, `rank_bm25`) over the same chunks, pre-filtered by the same product/reason-code/date metadata. `amdra eval --retrieval-only` scores the reason-code query alone (recall@1, recall@k, MRR) against all 42 labeled cases, bypassing the graph/LLM for a free A/B comparison. **Finding:** on this corpus hybrid shows no measurable lift over dense-only (`recall_at_1=0.357`, `mrr=0.655` either way) — every rank-1 miss is a same-reason-code *sibling* section (e.g. §3.2 outranking §3.3, §2.2 outranking §2.1/§2.3) that both lexical and semantic search score similarly, since disambiguating them requires the case's facts, not better query matching. `recall_at_k` is already 1.000, and the system is designed for the LLM to pick the right sibling from the retrieved set (the same mechanism the `verify` node's outcome-word check now enforces) — so query-side retrieval improvements have a real, structural ceiling here.
- **Vision-model fallback (M2d, done):** `gather_documents` routes to a new `vision_fallback` node (`llm.py` `ClaudeVisionReasoner`/`OfflineVisionReasoner`, `make_vision_reasoner`) whenever any receipt's `ocr_confidence` is below `settings.ocr_confidence_threshold` — a graph decision (`nodes.py` `route_after_documents`), never a model choice. The node reads the raw image via a new tool, `Toolbox.read_receipt_image` (`Scope.DOCS_READ`, distinct from `ocr_receipt`, which only ever returns text), sends it as an image content block to Claude, and re-derives `fact:amount`/`fact:delivery` from the (possibly recovered) result using the exact same formula `gather_documents` uses (factored into shared `_amount_fact`/`_delivery_fact` helpers so the two call sites can't drift). The resulting evidence stays `trusted=False`, same as programmatic OCR text. `OfflineVisionReasoner` is a deterministic stand-in used under `--offline`/tests: it always reports back the same lack of confidence, so the node's routing and evidence-merge wiring run every offline pass without ever calling an LLM, and severe-noise cases still escalate exactly as in M2c. The vision call's tokens/cost land in their own `llm_usage` entry, separate from `decide`'s. Not yet built: receipt-image embeddings (CLIP-style) for similar-receipt retrieval.

## 6. Reasoning (`llm.py`)

- **`ClaudeReasoner`:** `ChatAnthropic(...).with_structured_output(Recommendation, include_raw=True)` at temperature 0. It reads token usage from `usage_metadata` and computes cost from `config.PRICING_PER_MTOK` (verify the rates before trusting the cost numbers).
- **`OfflineReasoner`:** a rule-based baseline that reads computed facts and the *retrieved* policy text; for example, it parses the filing window from the POL-001 chunk. If a needed section wasn't retrieved, it escalates. It is the regression oracle, and the LLM is measured against it.

## 7. Verification and review

`verify` checks five things:

1. Every citation id exists.
2. Every quote is a verbatim, whitespace-normalized substring of that evidence item.
3. `policy_section` is one of the retrieved chunks.
4. The recommendation cites that chunk.
5. For `approve`/`deny` only: the cited section's own text says "approved"/"denied" — catches
   citing a sibling section (e.g. the approval rule) instead of the one whose condition is
   actually met. Skipped for `escalate`, which can legitimately cite a substantive section it
   couldn't conclusively apply (e.g. "no readable receipt total") rather than a section that
   itself mandates escalation.

If any check fails, the problems go back to `decide` as feedback, up to `max_decide_attempts` tries.

`review_gate` sends a case to a human when any of these is true:

- verification failed;
- the outcome is escalate;
- confidence is below 0.7;
- the outcome is **approve** (credits always need a human);
- the injection scanner flagged the case.

`human_review` runs behind `interrupt_before`. Provisional credit is issued only there: the node must hold the `credit:write` scope, and state must contain `human_decision.approved`. The credit is the difference for an amount mismatch and the full amount otherwise.

## 8. Security

| Threat | Control |
|---|---|
| Prompt injection in OCR text, narrative, or a receipt image | spotlighting (untrusted blocks, HTML-escaped), system prompt rule (including `vision_fallback`'s and `investigate`'s own prompts), regex scanner run on vision/investigator output too → review flag, tools not LLM-selectable in `"fixed"` mode, verifier |
| Cross-account data access | `@authorized` binds every call to the dispute's account |
| Excessive agency | per-node scopes; `decide` has none; only `human_review` has `credit:write`; `investigate`'s scope excludes `credit:write` entirely (§below) |
| Hallucinated evidence | verbatim-quote verification + retry + review; `investigate`'s tool wrappers build `Evidence` from structured tool output, never from the model's own text (§below) |
| Stale or wrong policy | effective-date and product filters; version recorded in evidence; `retrieve_policy` runs deterministically and unconditionally even in `"react"` mode |
| Audit gaps | append-only `audit`, `tool_calls` (including denials), `llm_usage` channels; checkpointer; optional LangSmith tracing (below) |

**Tracing (M3a, done):** set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` (see `.env.example`) to get a full LangSmith trace of every graph run — every `ChatAnthropic` call (`decide`, and `vision_fallback`'s vision call) and the `StateGraph` invocation itself are auto-instrumented. `langsmith` is already installed as a required `langchain-core` dependency, so this needs no new package and no code change — it is a config-only capability, off by default. OpenTelemetry packages present in the environment are transitive via `chromadb` only and are not wired to LangGraph; LangSmith was chosen over hand-instrumenting OTel for that reason.

**ReAct investigator authz (M3c, done):** `investigate` replaces `gather_transactions`/`gather_documents`/`vision_fallback` only when `Settings.investigator == "react"` (default `"fixed"`; `AMDRA_INVESTIGATOR` env var) — an opt-in ablation, not a replacement for the deterministic pipeline, so the 100%-scoring offline baseline and every existing test are unaffected. `NODE_SCOPES["investigate"]` grants `TXN_READ | DOCS_READ | POLICY_SEARCH`, never `CREDIT_WRITE`. Each tool the model can call (`nodes.py` `_build_investigator_tools`) is a thin wrapper around the exact same `@authorized` `Toolbox` methods every other node uses, with `account_id` **closed over, not a parameter the model can set** — a prompt injection in tool output can't even phrase a cross-account request, since the schema has no slot for it; `@authorized` remains the enforced backstop regardless (probed in `evals/authz_probes.py`: `investigate_node_issues_credit`, `investigate_node_cross_account_txn`, `investigate_node_cross_account_receipt`, plus a control). Each wrapper also independently builds the same `Evidence` shape `gather_transactions`/`gather_documents` already build for that data type (factored into shared helpers — `_txn_evidence`, `_filing_fact`, `_similar_fact`, `_fraud_fact`, `_receipt_evidence` — used by both the fixed pipeline and the react tool wrappers) and appends it to a list the node returns; the model's own free text is never spliced into `Evidence` or a citation. The disputed transaction and the filing-window fact are always guaranteed regardless of what the model chose to investigate (mirroring what `gather_transactions` always fetches); reason-code-specific facts (similar transactions, fraud signals, receipt totals/delivery dates) are genuinely investigator-driven — if the model didn't gather what a fact needs, that fact is honestly absent, same as a missing fact today. Loop mechanics use `langgraph.prebuilt.create_react_agent` (deprecated in langgraph 1.0 in favor of `langchain.agents.create_agent`, which needs the separate `langchain` package — not added; the prebuilt helper is still functional), bounded by `Settings.investigator_max_steps` (default 8) via `recursion_limit`. **Verified, not assumed:** `create_react_agent`'s `ToolNode` only auto-catches `ToolInvocationError` (malformed args) and re-raises everything else — the first live run (see M3c in §10) confirmed this the hard way, when a hallucinated transaction id's `KeyError` crashed 12 of 14 cases instead of coming back to the model as a recoverable error. Each tool wrapper now explicitly catches `KeyError` and returns it as text; `AuthorizationError` is deliberately left uncaught, since it can never legitimately fire (the closure design above means the model can never supply a bad `account_id`) and should stay the fatal wiring-bug signal every other node treats it as.

Planned (M4): an LLM-based injection classifier as a second opinion, canary tokens, and output PII redaction.

## 9. Evaluation (`evals/runner.py`)

| Dimension | Metric |
|---|---|
| Decision accuracy | overall, per scenario, confusion matrix; policy_section_accuracy |
| Retrieval | retrieval_recall (labeled section retrieved), retrieval_recall_at_1, retrieval_mrr (mean reciprocal rank), policy_version_accuracy |
| Citation faithfulness | citation_validity (verbatim), verification_pass_rate |
| Prompt-injection resistance | injection_resistance (correct and never approve), injection_detection, false_flag_rate |
| Tool authorization | red-team probe pass rate (10 probes, including over-blocking controls), unapproved_side_effects = 0 |
| Latency / cost | p50/p95 per case, tokens, total and per-case USD |

Results go to `evals/results/<timestamp>.{json,md}`. The offline baseline scores 100% on the current suite by construction, so the harness is proven before Claude is measured against it.

Next steps for the evals:

- Harder cases: conflicting evidence, blurry or rotated receipts, multilingual receipts, and injections phrased to evade the regex.
- An LLM-as-judge rationale-quality score, calibrated against human labels.
- Repeated runs to measure variance.

## 10. Milestones

- **M1 (done in this skeleton):** data generator, tools + authz, retrieval with filters, graph with verify/retry/HITL, offline baseline, eval harness, tests.
- **M2 (multimodal depth, done):**
  - retrieval eval set (**done**): `retrieval_mrr` and `retrieval_recall_at_1` in the eval harness, scored directly off the existing 42 labeled cases (no new query-label file needed);
  - hybrid retrieval with reranking (**done**, see §5 — no measurable lift on this corpus, a structural finding rather than an implementation gap);
  - noisy receipt rendering + OCR confidence scores (**done**, see §4/§5 — 4 new scenarios, tag `ocr_noise`);
  - a vision-model fallback for low-confidence OCR (**done**, see §2/§5 — `vision_fallback` node, graph-triggered, first LLM call outside `decide`).
- **M3 (agentic depth):**
  - LangSmith tracing (**done**, see §8 — config-only, no new dependency);
  - persistent checkpointer (**done**): `amdra run <id> --review --checkpoint-db PATH` uses `langgraph-checkpoint-sqlite`'s `SqliteSaver` instead of the default `MemorySaver`, so a human-review pause survives a process restart. Opt-in only — tests and `amdra eval` keep `MemorySaver`, so they stay fast and side-effect-free;
  - ReAct investigator subgraph with bound tools behind the same authz (**done**, see §2/§8 — `Settings.investigator = "react"`, opt-in ablation against the `"fixed"` default; offline-tested with a scripted fake tool-calling model, `tests/test_graph.py::test_investigate_builds_evidence_from_tool_calls_not_narration`). **Live comparison** (`claude-sonnet-4-5`, same 14 cases, one per scenario): `"react"` scored **1.000 decision accuracy (14/14)** vs `"fixed"`'s **0.857 (12/14)** — `"react"` correctly escalated both `*_noisy_severe` cases that `"fixed"` got wrong (see the `decide`-prompt finding below), at a modest cost premium (`$0.017`/case vs `$0.015`) but ~2.5x the latency (p50 16.2s vs 6.5s — the extra tool-call round trips). This first live run also caught two real bugs, now fixed: the investigator's task prompt never actually told the model the disputed transaction's `txn_id` (it had no way to call `get_transaction` correctly), and `create_react_agent`'s default error handling only catches `ToolInvocationError` (malformed args), not arbitrary exceptions like a `KeyError` from a hallucinated id — so `_build_investigator_tools`'s wrappers now explicitly catch `KeyError` and return it as text the model can react to, while `AuthorizationError` is deliberately left uncaught (see §8);
  - parallel gather nodes (recommended deprioritized — `gather_transactions`/`gather_documents` have a real data dependency and are both near-instant deterministic steps; the whole measured latency budget is dominated by the `decide` LLM call, not these two, so the payoff doesn't justify the fan-out/join complexity. Revisit only if profiling says otherwise).
- **M4 (robustness):**
  - adversarial injection suite (obfuscated, multilingual, image-only);
  - LLM classifier;
  - cost and latency budgets with Haiku routing for easy cases.
- **M5 (presentation):** reviewer UI (Streamlit) showing the evidence graph, citations, and audit trail, plus an eval dashboard.

## 11. Open questions

- Should `escalate` count as correct when the expected outcome is deny but the evidence is ambiguous? Today it is strict equality.
- Should the model see computed facts, raw records, or both? Computed facts raise accuracy, but they move reasoning out of the model, so an ablation is worth running.
- Should model routing (Haiku vs. Sonnet) depend on reason code or on confidence?
- **`decide`'s system prompt doesn't tell Claude what to do when a fact is unreadable/missing** — found by the first live run to actually exercise M2c's severe-noise scenarios with real Claude (§10, M3c): `"fixed"` mode scored only 0.857 decision accuracy on a 14-case run, getting *both* `*_noisy_severe` cases wrong (denied/approved instead of escalating), even though the evidence honestly says "Receipt total unknown... Difference: unknown." `OfflineReasoner` has this as an explicit hardcoded rule (`if receipt_total_cents is None: escalate`); `ClaudeReasoner`'s `SYSTEM_PROMPT` (`llm.py`) has no equivalent instruction, so Claude apparently reasons past the gap instead of treating "unreadable" as grounds to escalate. `"react"` mode scored 1.000 on the same 14 cases — worth investigating why (possibly: actively calling the OCR tool and seeing the garbled text firsthand makes the model more cautious than reading a terse "unknown" in a computed-fact string) before assuming `"react"` alone is the fix; the more direct fix is likely an explicit prompt rule. Not yet fixed — flagging for a follow-up pass.
- M3c `"react"` mode: `OfflineReasoner` still isn't guaranteed to handle every reason code gracefully against react-gathered evidence (e.g. a missing `fact:similar`/`fact:fraud` on a `DUPLICATE`/`FRAUD` case would raise a `KeyError`, caught by `decide`'s existing exception handling and routed to human review — safe, but not the same honest "no readable X" rationale the `AMOUNT_MISMATCH`/`NOT_RECEIVED` paths have). This combination isn't the intended production path — `"react"` mode is meant to be run with `ClaudeReasoner` — but it's worth being explicit that it isn't bulletproofed for every reason code the way the fixed pipeline is.
