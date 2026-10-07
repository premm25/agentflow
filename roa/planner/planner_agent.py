"""Planner Agent = Understanding Agent (intent) + Flow selection + Planner.

    understand  ->  select flow (Flow Registry, deterministic, recorded with its reason)  ->  plan inside that flow

Both LLM sub-steps are driven by files in harness/ (prompts, JSON schemas, agent registry, flow registry, guardrails,
fallback rules). The flow is selected by the registry before the planner runs, so the planner only chooses agents inside
it; when the registry cannot decide (no intent, no topic keywords) the planner proposes a flow exactly as before and the
plan guardrails decide. Failure never triggers autonomous re-planning: the fallback chain is LLM -> deterministic keyword
match -> human in the loop.
"""

import json
import time
from dataclasses import dataclass, field

from roa import llm, state, telemetry
from roa.harness import guardrails
from roa.harness.loader import HarnessBundle
from roa.harness.store import GuardedStore
from roa.models import CaseUnderstanding, FlowSelection, GuardrailEvent, Plan, TimeHorizon

PRINCIPAL = "planner"


@dataclass
class PlanOutcome:
    plan: Plan | None
    events: list[GuardrailEvent] = field(default_factory=list)
    reason: str = ""


def _log(store: GuardedStore, case_id: str, event: str, **fields):
    store.log(PRINCIPAL, f"cases/{case_id}/planner/log.jsonl", event, **fields)


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def flow_for_agents(bundle: HarnessBundle, agent_ids: list[str], hint: str | None = None) -> str:
    """Pick the flow a deterministic (non-LLM) agent selection belongs to (the Flow Registry decides)."""
    return bundle.flow_registry().flow_for_agents(agent_ids, hint)


def selection_for(bundle: HarnessBundle, flow_id: str, intent: str | None, source: str, reason: str,
                  matched: dict | None = None) -> FlowSelection:
    """A SELECTED FlowSelection for a registered flow (unknown ids are recorded as UNRESOLVED, never invented)."""
    reg = bundle.flow_registry()
    spec = reg.get(flow_id)
    if spec is None:
        return FlowSelection(status="UNRESOLVED", intent=intent, source="unresolved", reason=f"flow '{flow_id}' is not registered",
                             matched=matched or {})
    return FlowSelection(status="SELECTED", flow_id=spec.flow_id, name=spec.name, version=spec.version, description=spec.description,
                         intent=intent, source=source, reason=reason, matched=matched or {},
                         participating_agents=list(reg.task_order(spec)))


async def understand_case(bundle: HarnessBundle, store: GuardedStore, case_id: str) -> CaseUnderstanding:
    cs = state.get(case_id)
    case = cs.case
    max_calls = bundle.guardrails["budgets"]["max_llm_calls_per_case"]
    with telemetry.span("planner_agent.understanding", "AGENT", **{"roa.case_id": case_id}) as sp:
        t0, source = time.monotonic(), "llm"
        try:
            data, meta = await llm.call_structured(
                case_id, "understanding", bundle.model_for("understanding"), bundle.understanding_prompt,
                f"Case description:\n{case.description}", bundle.understanding_schema, max_calls=max_calls,
                timeout=bundle.manifest["llm"]["timeout_s"], **bundle.llm_opts("understanding"))
            u = CaseUnderstanding.model_validate(data)
            _log(store, case_id, "understanding", model=meta.model, output=u.model_dump(mode="json"))
        except llm.LLMError as e:
            fb = bundle.fallback["understanding_failure"]
            u = CaseUnderstanding(summary=case.description[:300], time_horizon=TimeHorizon.UNKNOWN)
            source = "fallback"
            sp.set_attribute("roa.fallback", fb["action"])
            _log(store, case_id, "understanding_fallback", action=fb["action"], error=str(e)[:300])
        if case.property_name:
            u.property_name = case.property_name  # explicit intake value beats an LLM guess
        state.mutate(case_id, lambda s: setattr(s, "understanding", u))
        sp.set_attribute("roa.intent", u.case_type_hint or "")
        state.record_event(case_id, "intent_resolved", intent=u.case_type_hint, horizon=u.time_horizon.value, source=source,
                           duration_ms=_ms(t0))
        return u


def select_flow(bundle: HarnessBundle, store: GuardedStore, case_id: str) -> FlowSelection:
    """Intent + keyword evidence -> a registered flow (deterministic). The decision is stored in the case state, logged, and
    emitted as a span, so it is an explicit runtime fact rather than something inferred from the plan afterwards."""
    cs = state.get(case_id)
    intent = cs.understanding.case_type_hint if cs.understanding else None
    with telemetry.span("planner_agent.flow_resolution", "CHAIN", **{"roa.case_id": case_id}) as sp:
        t0 = time.monotonic()
        res = bundle.resolve_flow(intent, cs.case.description)
        if res.resolved:
            sel = selection_for(bundle, res.flow_id, res.intent, res.source, res.reason, res.matched)
        else:
            sel = FlowSelection(status="UNRESOLVED", intent=res.intent, source="unresolved", reason=res.reason, matched=res.matched)
        state.set_flow_selection(case_id, sel)
        state.record_event(case_id, "flow_resolved", status=sel.status, flow_id=sel.flow_id, source=sel.source, duration_ms=_ms(t0))
        for k, v in {"roa.intent": sel.intent, "roa.flow.status": sel.status, "roa.flow.id": sel.flow_id, "roa.flow.name": sel.name or None,
                     "roa.flow.version": sel.version or None, "roa.flow.source": sel.source, "roa.flow.reason": sel.reason[:300],
                     "roa.flow.agents": json.dumps(sel.participating_agents)}.items():
            if v is not None:
                sp.set_attribute(k, v)
        _log(store, case_id, "flow_resolved", status=sel.status, flow_id=sel.flow_id, source=sel.source, reason=sel.reason,
             matched=sel.matched)
        return sel


def _adopt_flow(bundle: HarnessBundle, store: GuardedStore, case_id: str, flow_id: str, source: str, why: str) -> None:
    """The registry could not decide, and the planner's plan for this flow passed every guardrail: record that flow as selected."""
    prev = state.get(case_id).flow_selection
    sel = selection_for(bundle, flow_id, prev.intent if prev else None, source,
                        f"{prev.reason if prev else 'Registry did not select a flow'}. {why}", prev.matched if prev else {})
    state.set_flow_selection(case_id, sel)
    _log(store, case_id, "flow_adopted", flow_id=flow_id, source=source)


async def plan_case(bundle: HarnessBundle, store: GuardedStore, case_id: str) -> PlanOutcome:
    cs = state.get(case_id)
    u = cs.understanding
    sel = cs.flow_selection
    selected = sel.flow_id if sel is not None and sel.status == "SELECTED" else None
    max_calls = bundle.guardrails["budgets"]["max_llm_calls_per_case"]
    events: list[GuardrailEvent] = []
    reg = bundle.flow_registry()

    with telemetry.span("planner_agent.plan", "AGENT", **{"roa.case_id": case_id}) as sp:
        if selected:
            flows_text = (f"SELECTED FLOW (chosen by the flow registry; use it as case_type and plan only inside it):\n"
                          f"{reg.menu_line(selected)}")
        else:
            flows_text = f"FLOWS:\n{bundle.flow_menu()}"
        user = (f"Case description:\n{cs.case.description}\n\n"
                f"Understanding: {u.model_dump_json() if u else '{}'}\n\n"
                f"{flows_text}\n\nAGENT MENU:\n{bundle.agent_menu()}")
        llm_error = ""
        try:
            data, meta = await llm.call_structured(
                case_id, "planner", bundle.model_for("planner"), bundle.planner_prompt, user,
                bundle.planner_schema, max_calls=max_calls, timeout=bundle.manifest["llm"]["timeout_s"],
                **bundle.llm_opts("planner"))
            ordered, ev = _guard_plan(bundle, data["case_type"], data["agent_ids"], cs.case.description,
                                      u.case_type_hint if u else None, selected)
            events += ev
            _log(store, case_id, "plan_proposed", model=meta.model, raw=data, guardrails_passed=not guardrails.failed(ev))
            if not guardrails.failed(ev):
                sp.set_attribute("roa.plan.source", "llm")
                if selected is None:
                    _adopt_flow(bundle, store, case_id, data["case_type"], "planner",
                                "The planner proposed this flow and every plan guardrail passed.")
                return PlanOutcome(Plan(case_type=data["case_type"], agent_ids=ordered, reasoning=data["reasoning"], source="llm"), events)
            llm_error = "plan guardrails failed: " + "; ".join(f"{e.rule}: {e.detail}" for e in guardrails.failed(ev))
        except llm.LLMError as e:
            llm_error = str(e)
        _log(store, case_id, "planner_failed", error=llm_error[:400])
        state.record_event(case_id, "planner_fallback", error=llm_error[:300])

        # Fallback chain from harness/fallback/fallback.json: keyword match, then human.
        chain = bundle.fallback["planner_failure"]["chain"]
        if "keyword_match" in chain:
            matched = bundle.keyword_match(cs.case.description)
            if matched:
                case_type = selected or reg.flow_for_agents(matched, u.case_type_hint if u else None)
                ordered, ev = _guard_plan(bundle, case_type, matched, selected_flow=selected)
                events += ev
                if not guardrails.failed(ev):
                    sp.set_attribute("roa.plan.source", "keyword_fallback")
                    if selected is None:
                        _adopt_flow(bundle, store, case_id, case_type, "keyword_match", "Chosen from the agents whose keywords matched.")
                    _log(store, case_id, "plan_keyword_fallback", case_type=case_type, agent_ids=ordered)
                    return PlanOutcome(Plan(case_type=case_type, agent_ids=ordered,
                                            reasoning=f"Planner unavailable ({llm_error[:120]}); matched agent keywords.",
                                            source="keyword_fallback"), events)
        sp.set_attribute("roa.plan.source", "none")
        return PlanOutcome(None, events, reason=f"No valid plan could be produced: {llm_error[:300] or 'no agent keywords matched'}")


def _guard_plan(bundle: HarnessBundle, case_type: str, agent_ids: list[str], case_text: str | None = None,
                hint: str | None = None, selected_flow: str | None = None):
    with telemetry.span("guardrail.plan", "GUARDRAIL") as sp:
        ordered, ev = guardrails.check_plan(case_type, agent_ids, bundle, case_text, hint, selected_flow)
        bad = guardrails.failed(ev)
        sp.set_attribute("roa.guardrail.passed", not bad)
        if bad:
            sp.set_attribute("roa.guardrail.failed_rules", json.dumps([e.rule for e in bad]))
        return ordered, ev


async def run_planner_agent(bundle: HarnessBundle, store: GuardedStore, case_id: str) -> PlanOutcome:
    with telemetry.span("planner_agent", "AGENT", **{"roa.case_id": case_id}) as sp:
        u = await understand_case(bundle, store, case_id)
        sp.set_attribute("roa.understanding.hint", u.case_type_hint or "")
        sp.set_attribute("roa.understanding.horizon", u.time_horizon.value)
        select_flow(bundle, store, case_id)
        t0 = time.monotonic()
        outcome = await plan_case(bundle, store, case_id)
        sp.set_attribute("roa.plan.status", "ok" if outcome.plan else "needs_human")
        flow = state.get(case_id).flow_selection
        if flow is not None and flow.flow_id:
            sp.set_attribute("roa.flow.id", flow.flow_id)
        if outcome.plan:
            sp.set_attribute("roa.plan.case_type", outcome.plan.case_type)
            sp.set_attribute("roa.plan.source", outcome.plan.source)
            sp.set_attribute("roa.plan.agents", json.dumps(outcome.plan.agent_ids))
        state.add_guardrail_events(case_id, outcome.events)
        if outcome.plan is not None:
            state.mutate(case_id, lambda s: setattr(s, "plan", outcome.plan))
            state.record_event(case_id, "plan_ready", flow_id=outcome.plan.case_type, agents=outcome.plan.agent_ids,
                               source=outcome.plan.source, duration_ms=_ms(t0))
            _log(store, case_id, "plan_final", plan=outcome.plan.model_dump(mode="json"))
        else:
            state.record_event(case_id, "plan_failed", reason=outcome.reason[:300], duration_ms=_ms(t0))
        return outcome
