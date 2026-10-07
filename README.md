# AgentFlow

**Agentic Workflow Orchestration Platform**

AgentFlow is an agentic workflow orchestration platform developed by Prem Kumar as an engineering portfolio project. It demonstrates how multi-agent systems can combine planning, deterministic validation, governance, tool execution, and observability into a controlled workflow.

It covers multi-agent planning, deterministic validation, configuration quality checks, human-in-the-loop governance, and execution observability.

In practice, AgentFlow takes a support case, decides which flow and which worker agents should handle it, runs those agents one at a time, checks every result against a written contract, drafts a response from verified evidence only, and pauses for a person whenever something fails or before anything becomes final. Every step is recorded and shown in an execution timeline and a separate observability dashboard.

> **Status: a working proof of concept.** The four worker agents are **simulated**: their tools return deterministic placeholder data. Nothing is connected to a real revenue-management system or CRM, and nothing is ever sent to a client. The point of the project is the orchestration, validation and governance machinery around the agents, not the domain data.

The domain used for the demo is hotel revenue-operations support (pricing, overbooking, last room value, forecast). It is a fictional setup chosen to give the agents something concrete to reason about.

## What this repository demonstrates

The engineering areas shown here, with where each is explained below:

- **Multi-flow orchestration:** five flows (`pricing`, `overbooking`, `lrv`, `forecast`, `multi`), each with its own agents and limits, and the selection recorded for every case. See [Flow Registry](#flow-registry).
- **Flow Registry:** one typed registry over `harness/flows/flows.json`, validated against the agent and tool registries when the harness loads.
- **Intent-to-flow routing:** a deterministic, LLM-free step that picks the flow from the intent and the case text, and records why.
- **Planner / Orchestrator separation:** the planner decides *what* handles a case, inside the selected flow; a fixed LangGraph skeleton decides *how and when* it runs. See [Planner vs orchestrator](#planner-vs-orchestrator).
- **Four simulated specialist agents:** pricing, overbooking, last room value and forecast, each with a tool allow-list and a written evidence contract. Their data is placeholder data.
- **Deterministic validation:** seven code-level checks on every agent result before an optional LLM judge. See [Deterministic validation](#deterministic-validation).
- **Configuration / Rule Critic:** a pure check of the validation configuration itself.
- **Derived validation tests:** boundary cases generated from each agent's own contract.
- **Human-in-the-loop governance:** every failure, and the final response, waits for a person. See [Human-in-the-loop governance](#human-in-the-loop-governance).
- **Execution timeline and observability UI:** a per-case timeline built from recorded state, plus OpenTelemetry traces in a separate dashboard.
- **Automated testing:** 172 offline tests with a scripted fake LLM.

---

## Contents

1. [The problem it addresses](#the-problem-it-addresses)
2. [How a case flows](#how-a-case-flows)
3. [Architecture](#architecture)
4. [Planner vs orchestrator](#planner-vs-orchestrator)
5. [Flow Registry](#flow-registry)
6. [The four simulated agents and tool execution](#the-four-simulated-agents-and-tool-execution)
7. [Deterministic validation](#deterministic-validation)
8. [Config / Rule Critic and derived validation tests](#config--rule-critic-and-derived-validation-tests)
9. [Human-in-the-loop governance](#human-in-the-loop-governance)
10. [Execution timeline and observability](#execution-timeline-and-observability)
11. [Project structure](#project-structure)
12. [Run it locally](#run-it-locally)
13. [Run the tests](#run-the-tests)
14. [Example workflow](#example-workflow)
15. [Engineering and design decisions](#engineering-and-design-decisions)
16. [Limitations](#limitations)
17. [Future improvements](#future-improvements)
18. [Project note and attribution](#project-note-and-attribution)
19. [License](#license)

---

## The problem it addresses

LLM-driven agents are easy to demo and hard to trust. Typical failure modes, and what AgentFlow does about each:

| Problem | Approach in this project |
|---|---|
| Behaviour hidden in scattered prompts and code | A **harness**: agent specs, tool registry, flows, guardrails, validation policy and prompts live in versioned files under `harness/`, hash-locked at startup |
| Agents asserting things they did not look up | **Evidence is created by the harness** from recorded tool calls; an agent cannot cite a source it did not call |
| "Looks right" is decided by another LLM | **Deterministic checks decide first**; an LLM judge only covers what code cannot, and never overrides a failed check |
| A routing mistake silently sends a case to the wrong agents | An explicit **Flow Registry** selects the flow deterministically and records why; the planner plans *inside* it |
| Failures handled by autonomous retries and re-planning | Every failure **pauses for a person**: retry one agent, continue with a note, or abort |
| Configuration drift between agent contracts and the validator | A **Config Critic** checks the configuration itself, and **derived tests** generate boundary cases from each agent's declared contract |
| A black box once it is running | One OpenTelemetry trace per case, an **execution timeline** in the UI, and a separate dashboard |

## How a case flows

```
Case
 └─ intake: harness integrity check + input guardrails
     └─ Planner Agent
         ├─ Understanding Agent  (intent: topic hint, time horizon, property)
         ├─ Flow Registry        (deterministic intent -> flow selection, recorded with its reason)
         └─ Planner              (plans inside the selected flow; plan guardrails; fallbacks)
             └─ worker agents, ONE AT A TIME, in the flow's order
                 └─ validation layer  (deterministic checks, then an optional semantic judge, per agent)
                     └─ reporting layer  (report.json, report.md, response draft)
                         └─ final human approval  (approve / request changes / reject)
                             └─ finalize
```

Whenever a step fails, the case pauses at a **human-in-the-loop** point instead of guessing (see [Human-in-the-loop governance](#human-in-the-loop-governance)). A retry re-runs **only that agent**; earlier agents are not re-run.

## Architecture

```
                  ┌──────────────────────── orchestrator process (:8100) ────────────────────────┐
  Browser  ─────► │ FastAPI  ─►  LangGraph skeleton (fixed)                                       │
  (UI + HIL)      │                intake → planner_agent → run_agent* → validate → report →      │
                  │                hil_final → finalize        (* one agent per step, sequential) │
                  │                                                                               │
                  │  reads (immutable)             writes (guarded)            emits              │
                  │  harness/  ◄── roa.harness ──► runtime/ via GuardedStore ── OpenTelemetry ──┐ │
                  │  registries, flows,            logs, results, memory,                       │ │
                  │  guardrails, prompts…          validation logs, reports                     │ │
                  └──────────────────────────────────────────────────────────────────────────────┼─┘
                                     LLM gateway (OpenAI-compatible)                             │ OTLP/HTTP
                                                                                                 ▼
                  ┌──────────────── dashboard process (:8200) ────────────────┐
                  │ OTLP receiver → sqlite spans → traces, waterfall,          │  ◄── also embedded in the
                  │ agent trajectory, aggregate metrics                        │      orchestrator UI (iframe)
                  └────────────────────────────────────────────────────────────┘
```

- The **LangGraph skeleton is fixed in code** (`roa/graph.py`). Everything it consults (which agents exist, which may run for which case, what a guardrail allows, what counts as complete) is read from `harness/`.
- The graph carries only routing state. Business data lives in a sqlite case store (`roa/state.py`) that is written as soon as something changes, so the UI shows live status independent of checkpoint timing.
- The **dashboard is a separate process**; the orchestrator only exports OTLP to it. Any OTLP backend could replace it.
- Everything runs in **one process with sequential agents**. There is no agent-to-agent communication.

### The harness: "immutable but writable"

Definitions under `harness/` are immutable; logs and memory under `runtime/` are writable. `roa/harness/store.py` enforces it: `GuardedStore` refuses any write that resolves outside `runtime/`, a principal writes only inside its own subtree, `*.jsonl` and `*.log` are append-only, other files are write-once, and every refused attempt is audited. The harness is hashed at startup, set read-only, and re-verified at every case intake; the hash is stamped on every report and trace.

This is an enforcement layer for trusted in-process code, **not a sandbox**: the principal is a string the caller passes, and the hash check *detects* tampering at the next case rather than preventing it.

## Planner vs orchestrator

The two have different jobs and are kept apart on purpose.

| | **Planner Agent** (`roa/planner/`) | **Orchestrator** (`roa/graph.py`, `roa/runner.py`) |
|---|---|---|
| Decides | *What* should handle this case: intent, flow, which agents | *How and when* it runs: order, timeouts, validation, human pauses, state |
| Uses an LLM | Yes (Understanding Agent and Planner), with deterministic fallbacks | No. It is a fixed LangGraph skeleton |
| Output | `{case_type, agent_ids, reasoning}` constrained by a JSON schema | Runs one agent per step, records every transition |
| Guarded by | Plan guardrails: known flow, registered and enabled agents, no duplicates, fits the selected flow, **topic-supported** by the case text | The harness: allow-lists, call budgets, hash checks, write guard |

If the LLM fails or its plan is blocked, a deterministic keyword match from the agent specs is tried; if that finds nothing, a person decides. The system never guesses a plan.

## Flow Registry

`harness/flows/flows.json` is the single registry of flows; `roa/harness/flows.py` is the typed API over it (`FlowSpec`, `FlowRegistry`, reached as `bundle.flow_registry()`). Each flow has an id, name, version, `enabled`, the `intents` that select it, its allowed and default agents, and `max_agents`.

| Flow | Intent | Agents (run order) | Used for |
|---|---|---|---|
| `pricing` | pricing | `pricing_agent`, `forecast_agent` | rate / BAR / rate-shopping questions |
| `overbooking` | overbooking | `overbooking_agent` | booking and inventory discrepancies |
| `lrv` | lrv | `lrv_agent`, `forecast_agent` | last room value and inventory control |
| `forecast` | forecast | `forecast_agent` | occupancy / revenue forecast questions |
| `multi` | (none) | all four | text that raises topics no single flow covers |

**Intent to flow** is deterministic and uses no LLM: (1) the intent's flow, if the agents named by keywords in the case text are all inside it; (2) otherwise the flow the keyword evidence points to, or the multi-topic flow when several topics are raised; (3) otherwise *unresolved*. The result and its reason are stored as `CaseState.flow_selection` and never recomputed, so it survives retries, human review and a restart. The planner then plans inside the selected flow, and the `matches_selected_flow` guardrail rejects a plan in any other flow.

## The four simulated agents and tool execution

Four worker agents, each an `async def run(ctx, inp)` in `roa/agents/`. **Their tools return deterministic placeholder data seeded by the case id**; they stand in for a revenue-management system and a CRM and are not production integrations.

| Agent | Purpose (simulated) | Tools |
|---|---|---|
| `pricing_agent` | Rate, occupancy and competitor context; rate configuration | `read_pricing_snapshot`, `read_occupancy`, `read_competitor_rates`, `read_rate_configuration` |
| `overbooking_agent` | Booking status and inventory | `read_booking_status`, `read_inventory` |
| `lrv_agent` | Last room value and its update history | `read_lrv`, `read_lrv_update_log` |
| `forecast_agent` | Occupancy forecast vs last year | `read_occupancy_forecast` |

Agents receive only an `AgentContext` (`roa/context.py`), the harness's gateway:

- `ctx.call_tool(name)` / `ctx.call_tools(a, b, c)` enforce the agent's allow-list and call budget, open a span, record the call (args, result, result hash, duration), and return the record. `call_tools` runs the **independent tools of one agent concurrently**; agents themselves stay sequential.
- `ctx.claim(text, value, call, confidence)` creates evidence **anchored to a recorded call**.
- `ctx.log(...)`, `ctx.memory()`, `ctx.remember(...)` are the agent's own log and `MEMORY.md`.

The runner (`roa/runner.py`) loads the agent entrypoint, runs it under a span with a timeout, turns any exception into a `FAILED` task (which pauses for a human), and writes the result to the agent's own write-once file. The pricing agent intentionally returns inconclusive, low-confidence evidence for long-horizon cases, which fails validation and reaches a human.

## Deterministic validation

"What is complete" is not in a prompt. Each agent's contract is in its `harness/agents/<id>/AGENT.md` (`expected_evidence`: required tools, minimum evidence, confidence floors). The validation layer (`roa/validation/`) checks it per agent:

| Check | Fails when |
|---|---|
| `status_done` | The agent did not finish with `DONE` |
| `required_tools_called` | A tool the contract requires was never (successfully) called |
| `min_evidence` | Fewer evidence items than the contract requires |
| `evidence_provenance` | An evidence item is not backed by a recorded tool call, or its values differ from the tool output |
| `numbers_grounded` | A number in a claim does not appear in that tool's output |
| `confidence_floor` | The agent's or an evidence item's confidence is below the floor |
| `allowed_tools_only` | A tool outside the agent's allow-list was used |

Only if no check fails does an optional **LLM judge** assess semantic relevance against the same declared contract. If the judge is unavailable the deterministic result stands and the verdict is flagged. A `FAIL` sends the case to a person.

## Config / Rule Critic and derived validation tests

Validation is only as good as its configuration, so the configuration is checked too.

**Config Critic** (`roa/validation/critic.py`): `critique_harness(bundle) -> list[Finding]` is a pure, deterministic check of the validation configuration. It covers what the harness loader does not: the `validation.json` check list against the checks the validator really runs (missing, unknown, duplicate), severity values, unknown keys, and each agent's `expected_evidence` (types, floors within [0, 1], `min_evidence` feasibility against `max_tool_calls`, unknown keys). Findings carry a severity, a code, a location and a message. It is **advisory**: it is an explicit callable and is not wired into startup or the graph, and it does not change how cases are validated.

**Derived validation tests** (`tests/derived_cases.py`, `tests/test_derive.py`): boundary cases are generated from each agent's own contract and run through the real deterministic checks, for example `min_evidence = 3` gives 3 items valid and 2 invalid, a 0.7 floor gives 0.7 valid and 0.69 invalid, one case per missing required tool, a tool outside the allow-list, a non-`DONE` status. Each case names the check it targets and any other check it is known to trip, and the validator's failures must equal that set exactly. This is test tooling, not part of the runtime.

## Human-in-the-loop governance

Where a person is involved, as actually implemented:

| Stage | Trigger | Human options |
|---|---|---|
| `input` | Input guardrail failed (too short or long, prompt-injection pattern) | `continue`, `abort` |
| `plan` | No valid plan (planner unsure, topic not supported, or invalid) | `continue_with_agents`, `abort` |
| `agent` | An agent failed, timed out, or hit a tool error | `retry`, `continue` (skip it), `abort` |
| `validation` | An agent's result failed validation | `retry:<agent_id>`, `continue_with_note`, `abort` |
| `report` (shown as **Response**) | Always, before anything is final | `APPROVED`, `CHANGES_REQUESTED` (bounded), `REJECTED` |

Decisions arrive through `POST /cases/{id}/hil` or the UI buttons; options not valid for the stage are rejected. Each interrupt sits in its own graph node, requests are idempotent, and a resume can only be claimed once. After a restart, a case waiting on a person resumes from its checkpoint; one that was mid-run is marked failed. There is no autonomous re-planning.

The response draft is written from **verified evidence only** and passes output guardrails (numbers must come from evidence, no forbidden promises or claimed actions, no internal terms); if the LLM draft breaks one, or the LLM is down, a clean template draft is used. Nothing is posted anywhere.

## Execution timeline and observability

Two views of the same run, from the same recorded facts:

- **Execution timeline** (orchestrator UI, `GET /cases/{id}/timeline`): built from the canonical case state by a pure function (`roa/timeline.py`): trigger, intake, intent, flow (and why it was selected), plan, orchestration, workers (with the tools each called and their results), validation (with failed checks), human steps, response and outcome, each with a status and a recorded duration, plus lifecycle events and a "now" banner while the case runs. The UI renders what the runtime recorded; it decides nothing.
- **Observability dashboard** (`:8200`, separate process): OpenTelemetry with **one trace per case** covering every stage, agent, tool call, LLM call, guardrail, validator and human wait. It offers a waterfall per case, an **agent trajectory** view (each agent's ordered tool calls checked against its contract, retries as extra attempts, validation verdicts), and aggregate metrics (latency by component, validation outcomes, human waits by stage, LLM usage by model). Human waits can last hours, so they are recorded as their own spans and never held open.

The orchestrator UI embeds the dashboard per case and links both ways.

## Project structure

```
harness/          IMMUTABLE definitions (hash-locked)        runtime/   writable run data (git-ignored)
  manifest.json     models per role, LLM options, wiring       cases/<id>/…       per-case logs, results, reports
  agents/           agents.json + <id>/AGENT.md specs          agents/<id>/MEMORY.md   per-agent memory
  tools/            tools.json                                 audit.log.jsonl        refused write attempts
  flows/            flows.json (the Flow Registry)
  guardrails/       guardrails.json                          roa/       the application (Python package)
  planner/          PLANNER.md + planner.schema.json           graph.py           the LangGraph skeleton
  understanding/    UNDERSTANDING.md + schema                  planner/           Planner Agent (+ Understanding Agent)
  fallback/         fallback.json                              agents/            the four simulated worker agents
  validation/       validation.json + JUDGE.md                 tools/             tool registry (placeholder data)
  reporting/        reporting.json + REPORT.md                 validation/        validator + Config Critic
                                                               reporting/         reporting layer
dashboard/        separate observability process               harness/           loader, Flow Registry API, guardrails, write guard
  server.py         OTLP receiver + APIs                       timeline.py        execution timeline builder
  trajectory.py     agent trajectory builder                   context.py         AgentContext (tools, logs, memory)
  static/index.html the UI                                     runner.py          runs an agent in-process
                                                               llm.py             OpenAI-compatible gateway client
web/index.html    orchestrator UI                              api.py             FastAPI service
evals/            golden cases + judge calibration           tests/     pytest suite (offline, fake LLM)
scripts/          start.ps1 / stop.ps1 (Windows)             docs/      ARCHITECTURE.md
```

The Python package is named `roa` and the environment variables use the `ROA_` prefix. These are kept from the project's origin to avoid a large, risky rename; they do not matter to how the system works.

## Run it locally

**Prerequisites:** Python 3.13+. For full behaviour, an OpenAI-compatible LLM gateway serving the models named in `harness/manifest.json` (default `gpt-oss-20b`); change the model names there for another gateway. Without a gateway the system still runs using the deterministic fallbacks configured in `harness/fallback/fallback.json` (keyword planning, template draft, deterministic-only validation), but the LLM-backed steps are skipped.

```bash
git clone https://github.com/premm25/agentflow.git
cd agentflow
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env  -- then set ROA_LLM_BASE_URL and ROA_LLM_API_KEY
```

Start the two processes (any OS):

```bash
python -m dashboard      # observability dashboard on :8200
python -m roa            # orchestrator + UI on :8100
```

On Windows, `scripts\start.ps1` and `scripts\stop.ps1` start and stop both in the background.

Then open **http://localhost:8100/ui/** (orchestrator and timeline) and **http://localhost:8200/** (dashboard).

### Docker Compose

```bash
cp .env.example .env        # set ROA_LLM_BASE_URL and ROA_LLM_API_KEY
docker compose up --build   # orchestrator + UI on :8100, dashboard on :8200
docker compose down         # stop (add -v to also delete case history and traces)
```

One image runs both processes. Case files, the database and traces live in named volumes; `harness/` is baked into the image and stays hash-locked. If the gateway runs on the host machine, use `host.docker.internal` instead of `localhost` in `ROA_LLM_BASE_URL`. The app has **no authentication**: do not expose it beyond a trusted network.

### Configuration

Settings are read from the environment or a `.env` in the project root (prefix `ROA_`). `.env` is git-ignored; only `.env.example` (placeholders) is tracked.

| Variable | Default | Meaning |
|---|---|---|
| `ROA_LLM_BASE_URL` | `http://localhost:4000/v1` | OpenAI-compatible gateway |
| `ROA_LLM_API_KEY` | (empty) | Gateway key. **Never commit it** |
| `ROA_HOST`, `ROA_PORT` | `0.0.0.0`, `8100` | Orchestrator bind |
| `ROA_OTLP_ENDPOINT` | `http://localhost:8200/v1/traces` | Where traces are exported |
| `ROA_DASHBOARD_URL` | `http://localhost:8200` | Dashboard link shown in the UI |
| `ROA_OTEL_ENABLED` | `true` | Turn trace export off |
| `ROA_HARNESS_DIR`, `ROA_RUNTIME_DIR`, `ROA_DATA_DIR` | `harness/`, `runtime/`, `data/` | Locations |
| `ROA_LOG_LEVEL` | `INFO` | Logging |

Per-role models and LLM options are in `harness/manifest.json`.

## Run the tests

```bash
python -m pytest -q
```

The suite (172 tests) is **offline**: it uses a scripted fake LLM and needs no gateway. It covers the Flow Registry, flow routing, the planner guardrails, validation, the Config Critic, the derived validation cases, the execution timeline (including UI logic), telemetry and the dashboard.

There is also an evaluation harness (`python -m evals.run`: 12 golden cases and a judge-calibration set, with release gates in `evals/thresholds.json`). It runs against a real LLM gateway, so it is not part of the offline suite, and its results are not reported here.

## Example workflow

Create a case (the UI's form does the same):

```bash
curl -s -X POST http://localhost:8100/cases \
  -H "Content-Type: application/json" \
  -d '{"description": "Why is my rate only 1000 for tonight at Hotel Aurora? Competitors seem much higher and it is a long weekend.", "property_name": "Hotel Aurora"}'
```

What happens, all visible in the **Timeline** tab of the case:

1. **Intake** checks the harness hash and the input guardrails.
2. **Intent**: the Understanding Agent reads the case as pricing, near-term.
3. **Flow**: the Flow Registry selects the `pricing` flow and records why.
4. **Plan**: the planner proposes `pricing_agent` inside that flow; plan guardrails accept it.
5. **Workers**: `pricing_agent` calls `read_pricing_snapshot`, `read_occupancy` and `read_competitor_rates` (simulated data) and returns evidence anchored to those calls.
6. **Validation**: the deterministic checks run; if one fails, the case pauses at the `validation` step for a person.
7. **Response**: a draft is written from the verified evidence and waits for human approval.
8. **Outcome**: after a decision (`POST /cases/{id}/hil` or the UI buttons) the case completes or ends rejected.

Open the dashboard for the trace waterfall and the agent trajectory of the same case.

## Engineering and design decisions

- **Sequential agents, no agent-to-agent calls.** Worker agents run one after another in a single process. This keeps execution order, failure handling and traces easy to reason about; a test guards it.
- **Async agents.** Agents are `async` functions, so I/O-bound tool calls inside one agent can overlap while the graph still advances one agent per step.
- **Humans handle failure, not a re-planner.** Recovery decisions belong to a person; the system never silently retries or re-routes.
- **Fixed skeleton, configurable behaviour.** The graph is code; agents, flows, guardrails, prompts and policy are files with a loader that cross-validates them at startup.
- **The quality plane does not trust the execution plane.** Evidence comes from harness-recorded tool calls, and the judge sees only the declared contract and the returned evidence.
- **Deterministic before LLM.** Cheap, explainable checks decide first; the LLM covers only what code cannot.
- **Explicit flow selection.** The flow is a recorded fact in the case state, not inferred later from agent names.
- **Observability records, it does not decide.** The timeline and dashboard only display what the runtime recorded; removing them would not change which flow a case follows.
- **Config is tested like code.** The Critic and the derived cases exist because a validator is only as good as its configuration.

## Limitations

- **The agents and tools are simulated.** Output is deterministic placeholder data, not real revenue-management or CRM data.
- **No authentication.** Anyone who can reach port 8100 can approve cases.
- **Single process and sqlite.** Fine for a proof of concept; only light concurrency has been exercised.
- **A restart does not resume mid-run cases.** They are marked interrupted; cases waiting on a human resume normally.
- **The Config Critic is advisory.** It is not called at startup or from the graph.
- **Immutability is not a sandbox.** Tampering is detected at the next case, not prevented.
- **The semantic judge is an LLM and can misjudge.** It only runs after the deterministic checks pass and never overrides a failed check; its accuracy depends on the model behind the gateway.
- **No historical-case retrieval or learning**, and no alerting.
- **The evaluation suite is small** (12 golden cases, 10 judge examples) and needs a real LLM gateway.
- **The setup scripts are Windows PowerShell;** other platforms use the two `python -m` commands or Docker Compose.

## Future improvements

- Authentication and role-based approval before any non-local deployment.
- Real tool adapters behind the existing tool registry (the registry, allow-lists and validation are designed for this), with the same evidence and provenance rules.
- Run the Config Critic at startup or in CI as a gate.
- Calibrate the judge prompt against the labeled evidence sets for the model in use; grow the golden and judge sets.
- Resume mid-run cases after a restart.
- Historical-case retrieval through a tool plus agent memory.
- Alerting on failures from the dashboard; a CI pipeline for the offline suite.

## Project note and attribution

AgentFlow evolved from an earlier prototype. The original orchestrator foundation (the harness, the Planner and Understanding agents, the validation and reporting layers, observability and evals) and the Docker setup come from that prototype, written by Karthik Nambiar and Shubham Jamdar. This public version has been substantially prepared and restructured by Prem Kumar as a portfolio project: the Config / Rule Critic and the derived validation tests, the multi-flow orchestration with the Flow Registry and intent-to-flow routing, the execution timeline and UI, and the public packaging, documentation and cleanup.

## License

No license has been chosen yet. Until one is added, the code is not licensed for reuse or redistribution.
