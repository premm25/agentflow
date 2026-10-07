# AgentFlow: Architecture Summary

**What it is.** A case-handling orchestrator for support cases. One process hosts a Planner Agent, four worker agents, a validation layer and a reporting layer. Every agent action is traced, every result is checked against a written contract, and a human decides whenever something fails. Nothing is sent to a client or written to a CRM automatically.

## How a case flows

```
Case -> intake guardrails -> Planner Agent (Understanding Agent, then Planner)
     -> worker agents, one at a time -> validation layer (per agent)
     -> reporting layer (report + client draft) -> human approval -> done
```

Any failure (blocked input, no valid plan, agent failure, failed validation) pauses the case for a person, who can retry one agent, continue with a note, or abort. There is no autonomous re-planning and agents never talk to each other.

## Multi-flow orchestration: Flow Registry, intent to flow, and the execution timeline

```
Case ─► intent resolution ─► Flow Registry ─► selected flow ─► planner ─► LangGraph ─► worker agent(s) ─► tools
        (understanding)      (deterministic)   (explicit state)  (inside    (fixed        (one at a time)
                                                                  the flow)  skeleton)
   ─► validation ─► human review (when needed) ─► response ─► outcome          every step is a recorded fact
```

**Flow Registry.** `harness/flows/flows.json` is the one registry of flows; `roa/harness/flows.py` is the typed API over it (`FlowSpec`, `FlowRegistry`), reached as `bundle.flow_registry()`. Each flow has an id, `name`, `version`, `enabled`, the `intents` that select it, its allowed and default agents, `max_agents`, and (for the multi-topic flow) `multi_topic`. The five flows that already existed are unchanged in what they allow; they gained the routing metadata. The loader refuses a registry where an intent selects two flows, and flows files without the new metadata still load with defaults.

| Flow | Intent | Agents (run order) | Used for |
|---|---|---|---|
| `pricing` | pricing | `pricing_agent`, `forecast_agent` | rate / BAR / rate-shopping; forecast may accompany |
| `overbooking` | overbooking | `overbooking_agent` | booking and inventory discrepancies |
| `lrv` | lrv | `lrv_agent`, `forecast_agent` | Last Room Value and inventory control |
| `forecast` | forecast | `forecast_agent` | occupancy / revenue forecast questions |
| `multi` | (none) | all four | text that raises topics no single flow covers |

**Intent to flow.** The understanding step resolves the *intent* (`case_type_hint`; an LLM, with a deterministic fallback). Then `bundle.resolve_flow(intent, text)` selects the flow with no LLM, in this order: (1) the intent's flow, if the agents named by keywords in the case text are all inside it; (2) otherwise the flow the keyword evidence points to, or the multi-topic flow when several topics are raised; (3) otherwise *unresolved*. The result, with its reason and the keywords that matched, is stored as `CaseState.flow_selection` (`FlowSelection`) and never recomputed: it survives human review, retries and a restart because it lives in the case state (sqlite), which every node reads, not in the graph's routing fields. Selection is deterministic for the same intent and text.

**Planner.** The planner runs after flow selection and plans *inside* that flow: its prompt names the selected flow, and the plan guardrail `matches_selected_flow` rejects a plan in any other flow (the existing keyword fallback then plans inside the selected flow). If the registry could not select a flow, the planner proposes one exactly as before, the existing guardrails (including topic support) decide, and an accepted flow is recorded as selected by `planner`; a human-chosen plan records the flow as selected by `human`. No second planner and no second registry exist.

**LangGraph and workers.** The graph skeleton in `roa/graph.py` is unchanged: one worker agent per step, in the flow's order; validation, human review and checkpointing behave as before. The orchestrator invokes a worker through the runner; workers stay plain Python and reach data only through allow-listed tools.

**Lifecycle events.** Two records describe what happened, from the same source: the case state (canonical, polled by the UI) and OpenTelemetry spans (observability, read by the dashboard). State carries `history` milestones (`intent_resolved`, `flow_resolved`, `plan_ready`, `workflow_started`, `draft_generated`, ...) next to the stage transitions, plus per-call LLM records (role, model, latency, tokens; never prompts or keys). Spans add `planner_agent.flow_resolution` and put `roa.flow.id` on the planner, agent, human-wait and root spans, and `roa.intent` on the root. Events in the timeline carry the case id and, once selected, the flow and intent. There is no QA or governance layer in this runtime, so none is shown.

**UI.** `GET /cases/{id}/timeline` builds the timeline from the case state (`roa/timeline.py`, a pure function): trigger, intake, intent, flow, plan, orchestration, workers, validation, human, response, outcome, each with a status, a recorded duration and drill-down data (why a flow was selected, the tools each worker called and their results, the failed checks, the human request and decision, retries with their reasons), plus a flat event list and a "now" banner while the case runs. The orchestrator UI shows it as the Timeline tab and polls it while the case is live; `GET /flows` feeds the Flows page; the dashboard trace list and trajectory show the same flow from the spans. The UI decides nothing: it renders what the runtime recorded, and the flow is read from the state, never inferred from agent names.

**Runtime vs observability.** Flow resolution, the plan guardrail and the planner's behaviour are runtime. The timeline endpoint, the spans and the dashboard only record and display; removing them would not change which flow a case follows.

**Adding a flow.** Add an entry to `harness/flows/flows.json` (`name`, `version`, `enabled`, `intents`, `allowed_agents`, `default_agents`, `max_agents`, `description`) and its id to the planner schema's `case_type` enum. The loader cross-checks agents, intents and the enum at startup. Existing agents and tools can be reused; a new agent is added as described under "Adding a new agent" in the README. The Flows page and the timeline pick the new flow up with no UI change.

## The harness (what makes it agentic and governable)

All behaviour is defined in files under `harness/`, not in prompts scattered through code:

| Component | File(s) |
|---|---|
| Agent registry and per-agent specs (purpose, tools, expected evidence) | `agents/agents.json`, `agents/<id>/AGENT.md` |
| Tool registry | `tools/tools.json` |
| Flow registry (flows, their intents, agents and order; which agents may run for which case type) | `flows/flows.json` |
| Guardrails (input, plan, agent, output, budgets) | `guardrails/guardrails.json` |
| Planner and understanding schemas and prompts | `planner/`, `understanding/` |
| Fallback rules | `fallback/fallback.json` |
| Validation and reporting policy | `validation/`, `reporting/` |

The harness is hash-locked: it is read-only on disk, its hash is verified when each case starts, and the hash is stamped on every report and trace. Agents write only their own logs and `MEMORY.md` under `runtime/`, through one guarded write path that refuses anything else and audits every refusal.

## How "complete" and "correct" are decided

Not by an LLM guessing. Each agent's contract (required tools, minimum evidence, confidence floors) is in its `AGENT.md`. The validation layer checks it deterministically: required tools were called, every piece of evidence traces to a tool call the harness recorded, every number in a claim appears in that tool's output, confidence is above the floor. Only then does an LLM judge assess semantic relevance, against the same declared contract. Agents cannot invent evidence: the harness creates it from tool calls.

## Harness Config Critic (advisory)

`roa/validation/critic.py` exposes `critique_harness(bundle) -> list[Finding]`, a deterministic consistency check of the validation configuration. It is pure (no I/O, no LLM, no state), reads only the loaded `HarnessBundle`, and reports every problem in one call. It is an explicit callable: nothing calls it at startup or from the graph, and it does not change how any case is validated.

It complements `load_harness`, which already rejects broken tool, flow and agent references, and does not repeat those checks. It covers what the loader never inspected:

- `validation.json`: the `deterministic_checks` list against the checks `_deterministic` actually runs (missing, unknown, duplicate); one severity per check, valued `fail` or `warn`; the `judge` block; unknown top-level keys.
- Each `AGENT.md`: `expected_evidence` types, ranges (floors in [0, 1], integer `min_evidence` of at least 1), feasibility against `max_tool_calls`, unknown keys; `semantic_check`, `enabled`, `handles`, `max_tool_calls`, `timeout_s` types (the runtime uses them as flags, in `.lower()` / `', '.join`, and inside `min()` against the guardrail budgets); `semantic_check` while the judge is disabled; `allowed_unknowns`, which nothing reads.

Findings carry a severity (`error` or `warn`), a code, a location and a message. On the real harness the only finding is a `warn`: `severity.judge` is never read.

Known limits: the list of check names is duplicated in the critic, and a test compares it with what `_deterministic` emits, so a new runtime check fails that test until the critic is updated. The critic reports; it does not block.

## Derived validation cases (test tooling)

`tests/derived_cases.py` derives boundary cases from an agent's declared contract, as plain data, and `tests/test_derive.py` runs them through the real `_deterministic` checks: `min_evidence` = 3 gives 3 items valid and 2 invalid; a 0.7 floor gives 0.7 valid and 0.69 invalid; one case per required tool being missing; a tool outside the allow-list; an agent status other than DONE. Each case names the check it targets and declares any other check it is known to trip (removing a single-tool agent's only required tool also leaves no evidence), and the run requires the validator's failures to equal that set exactly. The expectations come from an independent restatement of the semantics, which a test compares with the real validator over a grid of scenarios. It supplements `tests/test_validation.py`; it does not replace it, and it is not part of the runtime.

## Observability

OpenTelemetry, one trace per case, covering every stage, agent, tool call, LLM call, guardrail, validator, report step and human wait. A separate dashboard process (port 8200) receives the traces and shows a waterfall per case plus aggregate metrics: latency by component, validation verdicts and failed checks, human-wait time by stage, and tokens by model. Human waits can last hours, so they are recorded as their own spans and never held open. Nothing about tracing lives inside the agents.

### Agent trajectory

The dashboard's **Trajectory** view shows the path each case actually took: the Planner Agent (topic hint, horizon, plan and its source), then each worker agent run with its ordered tool calls checked against the agent's declared contract (required tools called, missing, extra), each validation verdict with the failed checks, and every human decision. Retries appear as extra attempts. Summary metrics: agent runs and retries, tool correctness, missing and extra tool calls, first-try validation pass rate, human interventions and wait time. It is rebuilt from the same OpenTelemetry spans, so it needs nothing beyond the trace.

The orchestrator UI and the dashboard are connected both ways: each case in the orchestrator UI has Case / Trajectory / Trace buttons that embed the live dashboard in place (plus a new-tab link and an overall Dashboard button), and the dashboard links back with "Open case in orchestrator". Deep links work in both directions (`/ui/#case=<id>`, `:8200/#trace=<id>&tab=trajectory`).

## Evaluation

`python -m evals.run` runs 12 golden cases through the real graph and models: routing, human-pause behaviour, validation outcomes, grounding, draft safety, and adversarial inputs (prompt injection, oversized input, unroutable cases). It also calibrates the LLM judge against 10 labeled evidence sets (false-pass and false-block rates). Release gates are in `evals/thresholds.json`; the run exits non-zero if any is missed.

The suite is small, so treat it as a baseline for catching regressions, not as proof of quality. Results depend on the model served by the gateway and are not reported here.

## Latency

Pipeline overhead is small; time is dominated by LLM decoding on the gateway. Design choices, all in the harness manifest and code:

- All four LLM roles use `gpt-oss-20b` at `reasoning_effort: low`, with a per-role output-token cap so no call can run away.
- One keep-alive HTTP connection pool per event loop.
- Independent work runs concurrently: validator judge calls across agents, and independent tool calls inside one agent. Worker agents stay strictly sequential (a test guards this).

Lower reasoning effort made the planner more likely to guess an agent for a vague case, so the planner prompt has an explicit abstain rule and a deterministic guardrail blocks any plan not backed by the case text or the understanding agent's independent topic hint. Safety cases (prompt injection, oversized input, unroutable case) are marked critical in the evals: an average can never hide a miss.

## Deliberate limits (honest list)

- Tools are placeholders with deterministic data. Real revenue-management and CRM tools would plug in through the tool registry.
- Immutability is enforcement for in-process code, not a sandbox; tampering is detected at the next case, not prevented.
- No API authentication yet. Single process, sqlite: fine for a POC, not for load.
- No historical-case retrieval yet. The agent `MEMORY.md` files and a retrieval tool are the intended place for it.
- The Config Critic is not called at startup or from the graph yet (it is an explicit, advisory callable).
- A server restart marks running cases as interrupted; cases waiting on a human resume normally.
