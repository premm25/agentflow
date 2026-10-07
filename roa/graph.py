"""The orchestration skeleton (fixed in code; everything it consults lives in harness/ files):

  intake -> planner_agent -> run_agent (sequential, one per step) -> validate -> report -> final approval -> finalize

There is no autonomous re-plan. Every failure (blocked input, unroutable plan, agent failure,
failed validation) pauses the case for a human, who can retry one agent, continue, or abort.
Each interrupt sits in its own node and every side effect that must not repeat happens after
`interrupt()` returns, because LangGraph re-runs the whole node from the top on resume.
"""

from functools import wraps
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from roa import state, telemetry
from roa.harness import guardrails
from roa.harness.loader import HarnessBundle, HarnessTampered
from roa.harness.store import GuardedStore
from roa.models import CaseStatus, GuardrailEvent, HILDecision, HILRequest, Plan, TaskStatus
from roa.planner import flow_for_agents, run_planner_agent, selection_for
from roa.reporting import draft_response, revise_response, write_report
from roa.runner import run_agent
from roa.validation import pending_failures, run_validation_layer


class GraphState(TypedDict, total=False):
    case_id: str
    queue: list[str]
    to_validate: list[str]
    last_agent: str
    last_status: str
    plan_ok: bool
    input_blocked: bool
    final: Optional[str]  # COMPLETED | ABORTED | FAILED, set by whichever node ends the case
    final_reason: Optional[str]


def validate_human_decision(bundle: HarnessBundle, req: HILRequest, decision: str, agent_ids: list[str] | None) -> str | None:
    """Returns an error message if the human's choice is not allowed for this request."""
    if req.stage == "plan" and decision == "continue_with_agents":
        if not agent_ids:
            return "choose at least one agent"
        ordered, ev = guardrails.check_plan(flow_for_agents(bundle, agent_ids), agent_ids, bundle)
        bad = guardrails.failed(ev)
        return "; ".join(f"{e.rule}: {e.detail}" for e in bad) if bad else None
    if decision not in req.options:
        return f"decision '{decision}' not allowed; options: {req.options}"
    return None


def build_graph(checkpointer, bundle: HarnessBundle, store: GuardedStore):
    max_revisions = bundle.step_policy["max_report_revisions"]

    def olog(case_id: str, event: str, **fields):
        store.log("orchestrator", f"cases/{case_id}/orchestrator/log.jsonl", event, **fields)

    def traced(fn):
        @wraps(fn)
        async def wrapper(gs: GraphState):
            cs = state.get(gs["case_id"])
            with telemetry.attach_case(cs.trace_id, cs.root_span_id):
                return await fn(gs)

        return wrapper

    async def human(case_id: str, build) -> HILDecision:
        """Pause for a human. Idempotent on re-run; the wait is emitted as its own span."""
        req = state.ensure_pending_hil(case_id, build)
        answer = interrupt({"hil_id": req.hil_id, "type": req.type, "stage": req.stage, "reason": req.reason,
                            "options": req.options})
        end_ns = telemetry.now_ns()
        d = HILDecision(hil_id=req.hil_id, type=req.type, stage=req.stage, decision=answer["decision"],
                        comment=answer.get("comment"), agent_id=answer.get("agent_id"),
                        agent_ids=answer.get("agent_ids") or [], waited_ms=int((end_ns - req.requested_at_ns) / 1e6))
        cs = state.get(case_id)
        telemetry.emit_span(cs.trace_id, cs.root_span_id, "hil.wait", req.requested_at_ns, end_ns, "CHAIN", **{
            "roa.case_id": case_id, "roa.hil.id": req.hil_id, "roa.hil.type": req.type, "roa.hil.stage": req.stage,
            "roa.hil.decision": d.decision, "roa.hil.agent_id": d.agent_id,
            "roa.flow.id": cs.flow_selection.flow_id if cs.flow_selection else None})
        state.resolve_hil(case_id, d)
        olog(case_id, "human_decision", hil_id=d.hil_id, stage=d.stage, decision=d.decision, comment=d.comment,
             agent_id=d.agent_id, waited_ms=d.waited_ms)
        return d

    def request(case_id: str, type_: str, stage: str, reason: str, options: list[str], agent_id: str | None = None,
                failing: list[str] | None = None, context: dict | None = None):
        def build(hil_id: str) -> HILRequest:
            return HILRequest(hil_id=hil_id, type=type_, stage=stage, reason=reason, options=options, agent_id=agent_id,
                              failing_agents=failing or [], context=context or {}, requested_at_ns=telemetry.now_ns())

        return build

    # ------------------------------------------------------------------ nodes
    @traced
    async def intake(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        cs = state.get(case_id)
        with telemetry.span("intake", "GUARDRAIL", **{"roa.case_id": case_id}) as sp:
            try:
                bundle.verify()
                harness_ev = GuardrailEvent(point="harness", rule="harness_immutable", passed=True,
                                            detail=f"hash {bundle.hash[:12]} verified")
            except HarnessTampered as e:
                state.add_guardrail_events(case_id, [GuardrailEvent(point="harness", rule="harness_immutable",
                                                                    passed=False, detail=str(e))])
                sp.set_attribute("roa.harness.tampered", True)
                return {"final": "FAILED", "final_reason": f"harness integrity check failed: {e}"}
            events = [harness_ev] + guardrails.check_input(cs.case.description, bundle)
            state.add_guardrail_events(case_id, events)
            bad = guardrails.failed(events)
            sp.set_attribute("roa.guardrail.passed", not bad)
            olog(case_id, "intake", guardrails_passed=not bad, failed=[e.rule for e in bad])
            return {"input_blocked": bool(bad)}

    @traced
    async def hil_input(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        bad = [e for e in state.get(case_id).guardrail_events if e.point == "input" and not e.passed]
        d = await human(case_id, request(case_id, "failure_review", "input",
                                         "Input guardrail failed: " + "; ".join(f"{e.rule}: {e.detail}" for e in bad),
                                         ["continue", "abort"]))
        if d.decision == "abort":
            return {"final": "ABORTED", "final_reason": "aborted by human at input guardrail"}
        state.mutate(case_id, lambda s: s.caveats.append("Input guardrail was overridden by a human reviewer."))
        return {"input_blocked": False}

    @traced
    async def planner_node(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        state.set_stage(case_id, CaseStatus.PLANNING)
        outcome = await run_planner_agent(bundle, store, case_id)
        if outcome.plan is None:
            olog(case_id, "plan_failed", reason=outcome.reason)
            return {"plan_ok": False, "final_reason": outcome.reason}
        olog(case_id, "plan_ready", plan=outcome.plan.model_dump(mode="json"))
        return {"plan_ok": True, "queue": list(outcome.plan.agent_ids), "to_validate": []}

    @traced
    async def hil_plan(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        reason = gs.get("final_reason") or "The planner could not produce a valid plan."
        options = ["continue_with_agents", "abort"]
        d = await human(case_id, request(case_id, "failure_review", "plan", reason, options,
                                         context={"available_agents": [a.agent_id for a in bundle.enabled_agents()]}))
        if d.decision == "abort":
            return {"final": "ABORTED", "final_reason": "aborted by human: no valid plan"}
        chosen = list(dict.fromkeys(d.agent_ids))
        ordered, ev = guardrails.check_plan(flow_for_agents(bundle, chosen), chosen, bundle)
        state.add_guardrail_events(case_id, ev)
        plan = Plan(case_type=flow_for_agents(bundle, chosen), agent_ids=ordered,
                    reasoning="Agents chosen by a human reviewer.", source="human")
        state.mutate(case_id, lambda s: setattr(s, "plan", plan))
        prev = state.get(case_id).flow_selection  # the registry's attempt stays visible in the reason
        state.set_flow_selection(case_id, selection_for(
            bundle, plan.case_type, prev.intent if prev else None, "human",
            "Flow inferred from the agents a human reviewer chose" + (f" (registry: {prev.reason})." if prev and prev.reason else "."),
            prev.matched if prev else {}))
        state.record_event(case_id, "plan_ready", flow_id=plan.case_type, agents=ordered, source="human")
        return {"plan_ok": True, "queue": ordered, "to_validate": [], "final_reason": None}

    @traced
    async def run_agent_node(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        queue = list(gs["queue"])
        agent_id = queue.pop(0)
        if not state.get(case_id).tasks:
            cs0 = state.get(case_id)
            state.record_event(case_id, "workflow_started", flow_id=cs0.flow_selection.flow_id if cs0.flow_selection else None,
                               agents=list(cs0.plan.agent_ids) if cs0.plan else [])
        state.set_stage(case_id, CaseStatus.EXECUTING)
        state.mutate(case_id, lambda s: s.waived_agents.remove(agent_id) if agent_id in s.waived_agents else None)
        task = await run_agent(bundle, store, case_id, agent_id)
        olog(case_id, "agent_finished", agent_id=agent_id, attempt=task.attempt, status=task.status.value)
        to_validate = list(gs.get("to_validate", []))
        if task.status == TaskStatus.DONE and agent_id not in to_validate:
            to_validate.append(agent_id)
        return {"queue": queue, "to_validate": to_validate, "last_agent": agent_id, "last_status": task.status.value}

    @traced
    async def hil_agent(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        agent_id = gs["last_agent"]
        task = state.latest_task(state.get(case_id), agent_id)
        reason = f"{agent_id} {task.status.value}: {(task.result.reason if task.result else None) or 'no detail'}"
        d = await human(case_id, request(case_id, "failure_review", "agent", reason, ["retry", "continue", "abort"],
                                         agent_id=agent_id))
        if d.decision == "abort":
            return {"final": "ABORTED", "final_reason": f"aborted by human after {agent_id} failed"}
        if d.decision == "retry":
            return {"queue": [agent_id] + list(gs.get("queue", []))}
        task.status = TaskStatus.SKIPPED
        state.upsert_task(case_id, task)
        state.mutate(case_id, lambda s: s.caveats.append(f"{agent_id} failed and was skipped by a human reviewer."))
        return {}

    @traced
    async def validate_node(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        state.set_stage(case_id, CaseStatus.VALIDATING)
        to_validate = gs.get("to_validate", [])
        verdicts = await run_validation_layer(bundle, store, case_id, to_validate) if to_validate else []
        olog(case_id, "validated", verdicts={v.agent_id: v.status for v in verdicts})
        return {"to_validate": []}

    @traced
    async def hil_validation(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        cs = state.get(case_id)
        failing = pending_failures(cs)
        detail = []
        for a in failing:
            v = state.latest_verdict(cs, a)
            why = ", ".join(f"{c.name} ({c.detail})" for c in v.checks if not c.passed) or (v.judge.rationale if v.judge else "")
            detail.append(f"{a}: {why}")
        options = [f"retry:{a}" for a in failing] + ["continue_with_note", "abort"]
        d = await human(case_id, request(case_id, "failure_review", "validation",
                                         "Validation failed for " + "; ".join(detail), options, failing=failing))
        if d.decision == "abort":
            return {"final": "ABORTED", "final_reason": "aborted by human after validation failure"}
        if d.decision.startswith("retry:"):
            return {"queue": [d.decision.split(":", 1)[1]]}
        note = d.comment or "no note given"
        state.mutate(case_id, lambda s: (s.waived_agents.extend(a for a in failing if a not in s.waived_agents),
                                         s.caveats.append(f"Human accepted failed validation for {', '.join(failing)}: {note}")))
        return {}

    @traced
    async def report_node(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        state.set_stage(case_id, CaseStatus.REPORTING)
        with telemetry.span("reporting_layer", "CHAIN", **{"roa.case_id": case_id}):
            await draft_response(bundle, store, case_id)
            await write_report(bundle, store, case_id)
        return {}

    @traced
    async def hil_final(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        cs = state.get(case_id)
        options = ["APPROVED"] + (["CHANGES_REQUESTED"] if cs.revision_count < max_revisions else []) + ["REJECTED"]
        d = await human(case_id, request(case_id, "final_approval", "report",
                                         "Review the response draft and report.", options))
        if d.decision == "APPROVED":
            return {"final": "COMPLETED", "final_reason": "approved by human"}
        if d.decision == "REJECTED":
            return {"final": "FAILED", "final_reason": "response rejected by human reviewer"}
        return {"final": None, "final_reason": d.comment or ""}

    @traced
    async def revise_node(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        state.set_stage(case_id, CaseStatus.REPORTING)
        with telemetry.span("reporting_layer.revision", "CHAIN", **{"roa.case_id": case_id}):
            await revise_response(bundle, store, case_id, gs.get("final_reason") or "")
            await write_report(bundle, store, case_id)
        return {"final_reason": None}

    @traced
    async def finalize(gs: GraphState) -> GraphState:
        case_id = gs["case_id"]
        final = gs.get("final") or "FAILED"
        reason = gs.get("final_reason")
        stage = {"COMPLETED": CaseStatus.COMPLETED, "ABORTED": CaseStatus.ABORTED}.get(final, CaseStatus.FAILED)
        state.set_stage(case_id, stage, reason)
        cs = state.get(case_id)
        if cs.understanding is not None:
            await write_report(bundle, store, case_id)
        olog(case_id, "finalized", stage=stage.value, reason=reason)
        telemetry.emit_root(cs.trace_id, cs.root_span_id, cs.started_ns, telemetry.now_ns(),
                            ok=(stage != CaseStatus.FAILED),  # a human abort is a valid outcome, not a system error
                            **{"roa.case_id": case_id, "roa.final_stage": stage.value, "roa.final_reason": reason,
                               "roa.harness.version": cs.harness_version, "roa.harness.hash": cs.harness_hash,
                               "roa.agents": ",".join(cs.plan.agent_ids) if cs.plan else "",
                               "roa.flow.id": cs.flow_selection.flow_id if cs.flow_selection else None,
                               "roa.intent": cs.understanding.case_type_hint if cs.understanding else None,
                               "roa.hil.count": len(cs.hil_decisions),
                               "roa.llm.calls": cs.llm_usage.calls,
                               "roa.llm.prompt_tokens": cs.llm_usage.prompt_tokens,
                               "roa.llm.completion_tokens": cs.llm_usage.completion_tokens})
        return {}

    # ------------------------------------------------------------------ routing
    def _ended(gs: GraphState) -> bool:
        return bool(gs.get("final"))

    def after_intake(gs):
        return "finalize" if _ended(gs) else ("hil_input" if gs.get("input_blocked") else "planner")

    def after_hil_input(gs):
        return "finalize" if _ended(gs) else "planner"

    def after_planner(gs):
        return "run_agent" if gs.get("plan_ok") else "hil_plan"

    def after_hil_plan(gs):
        return "finalize" if _ended(gs) else "run_agent"

    def after_run_agent(gs):
        if gs.get("last_status") in (TaskStatus.FAILED.value, TaskStatus.NOT_SUPPORTED.value):
            return "hil_agent"
        return "run_agent" if gs.get("queue") else "validate"

    def after_hil_agent(gs):
        if _ended(gs):
            return "finalize"
        return "run_agent" if gs.get("queue") else "validate"

    def after_validate(gs):
        return "hil_validation" if pending_failures(state.get(gs["case_id"])) else "report"

    def after_hil_validation(gs):
        if _ended(gs):
            return "finalize"
        return "run_agent" if gs.get("queue") else "report"

    def after_hil_final(gs):
        return "finalize" if _ended(gs) else "revise"

    g = StateGraph(GraphState)
    for name, fn in [("intake", intake), ("hil_input", hil_input), ("planner", planner_node), ("hil_plan", hil_plan),
                     ("run_agent", run_agent_node), ("hil_agent", hil_agent), ("validate", validate_node),
                     ("hil_validation", hil_validation), ("report", report_node), ("hil_final", hil_final),
                     ("revise", revise_node), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, ["finalize", "hil_input", "planner"])
    g.add_conditional_edges("hil_input", after_hil_input, ["finalize", "planner"])
    g.add_conditional_edges("planner", after_planner, ["run_agent", "hil_plan"])
    g.add_conditional_edges("hil_plan", after_hil_plan, ["finalize", "run_agent"])
    g.add_conditional_edges("run_agent", after_run_agent, ["hil_agent", "run_agent", "validate"])
    g.add_conditional_edges("hil_agent", after_hil_agent, ["finalize", "run_agent", "validate"])
    g.add_conditional_edges("validate", after_validate, ["hil_validation", "report"])
    g.add_conditional_edges("hil_validation", after_hil_validation, ["finalize", "run_agent", "report"])
    g.add_edge("report", "hil_final")
    g.add_conditional_edges("hil_final", after_hil_final, ["finalize", "revise"])
    g.add_edge("revise", "hil_final")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
