"""Intent -> flow selection through the real graph: the registry selects the flow, the planner plans inside it, the workers run,
validation and human review behave as before, and the decision is persisted and observable. The LLM is a fake gateway
(tests/flow_helpers.py); everything else is the real runtime."""

import asyncio
from dataclasses import replace

import pytest

from roa import state, telemetry
from roa.models import CaseStatus
from tests.conftest import MEM
from tests.flow_helpers import RoutedLLM
from tests.test_flows import answer, env, new_case, run, start

CASES = [  # (name, case text, flow, selection source, agents that must run)
    ("pricing", "Why is my rate only 1000 on 15 August at Hotel Aurora?", "pricing", "intent_match", ["pricing_agent"]),
    ("overbooking", "I received an overbooking notification for a booking I thought was confirmed.", "overbooking", "intent_match", ["overbooking_agent"]),
    ("lrv", "The last room value looks stuck. Is the forecast lock still active?", "lrv", "intent_match", ["lrv_agent", "forecast_agent"]),
    ("forecast", "The occupancy forecast for next week looks off compared to last year.", "forecast", "intent_match", ["forecast_agent"]),
    ("multi", "I got an overbooking notification for a booking I thought was confirmed, and the last room value looks stuck.",
     "multi", "multiple_topics", ["overbooking_agent", "lrv_agent"]),
]


@pytest.fixture()
def fake(bundle, monkeypatch):
    return RoutedLLM(bundle).install(monkeypatch)


def _events(cs, name):
    return [h for h in cs.history if h.get("event") == name]


# ---------------------------------------------------------------- every existing flow resolves, plans and executes
@pytest.mark.parametrize("name,text,flow,source,agents", CASES, ids=[c[0] for c in CASES])
def test_each_existing_flow_is_selected_planned_and_executed(tmp_path, bundle, fake, name, text, flow, source, agents):
    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cid = new_case(bundle, text)
            cs = await start(graph, cid)
            sel = cs.flow_selection
            assert (sel.status, sel.flow_id, sel.source) == ("SELECTED", flow, source)  # an explicit runtime record, not inferred afterwards
            assert sel.name and sel.version == "1.0.0" and sel.reason and sel.intent == fake._infer_hint(text)
            assert sel.participating_agents == list(bundle.flow_registry().task_order(bundle.flow_registry().get(flow)))
            assert cs.plan.case_type == flow and cs.plan.agent_ids == agents and cs.plan.source == "llm"  # the planner planned inside the flow
            assert [t.agent_id for t in sorted(cs.tasks, key=lambda t: t.started_at)] == agents  # the right workers, in order
            assert all(state.latest_verdict(cs, a) is not None for a in agents)  # validation still ran for each
            assert cs.pending_hil.type == "final_approval" and cs.stage == CaseStatus.WAITING_FOR_HUMAN  # and human approval is still required
            fr = _events(cs, "flow_resolved")
            assert len(fr) == 1 and fr[0]["flow_id"] == flow and fr[0]["source"] == source
            cs = await answer(graph, cid, "APPROVED")
            assert cs.stage == CaseStatus.COMPLETED and cs.flow_selection.flow_id == flow

    run(scenario())


def test_different_cases_follow_different_flows(tmp_path, bundle, fake):
    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            seen = {}
            for name, text, flow, _, _ in CASES:
                seen[name] = (await start(graph, new_case(bundle, text))).flow_selection.flow_id
            assert seen == {c[0]: c[2] for c in CASES} and len(set(seen.values())) == 5

    run(scenario())


# ---------------------------------------------------------------- determinism, fallbacks, unsupported intent
def test_selection_does_not_depend_on_the_llm_when_the_evidence_is_deterministic(tmp_path, bundle, fake):
    text = "I received an overbooking notification for a booking I thought was confirmed."
    fake.down = {"understanding", "planner"}

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cs = await start(graph, new_case(bundle, text))
            assert (cs.flow_selection.flow_id, cs.flow_selection.source, cs.flow_selection.intent) == ("overbooking", "keyword_match", None)
            assert "No intent was resolved" in cs.flow_selection.reason
            assert cs.plan.source == "keyword_fallback" and cs.plan.case_type == "overbooking" and cs.plan.agent_ids == ["overbooking_agent"]
            assert _events(cs, "intent_resolved")[0]["source"] == "fallback" and _events(cs, "planner_fallback")

    run(scenario())


def test_an_unregistered_intent_falls_back_to_the_keyword_evidence(tmp_path, bundle, fake):
    fake.hint = "billing"  # the understanding step returned an intent no flow is registered for

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cs = await start(graph, new_case(bundle, "Why is my rate so high compared to competitors?"))
            assert (cs.flow_selection.flow_id, cs.flow_selection.source) == ("pricing", "keyword_match")
            assert "not registered to any flow" in cs.flow_selection.reason and cs.plan.agent_ids == ["pricing_agent"]

    run(scenario())


@pytest.mark.parametrize("hint", [None, "billing"])
def test_an_unresolvable_case_is_unresolved_and_goes_to_a_human_exactly_as_before(tmp_path, bundle, fake, hint):
    fake.hint = hint
    fake.plan_override = {"case_type": "overbooking", "agent_ids": ["overbooking_agent"], "reasoning": "most likely overbooking"}  # a guess

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cid = new_case(bundle, "Something strange happened yesterday, can you please take a look at it soon?")
            cs = await start(graph, cid)
            assert (cs.flow_selection.status, cs.flow_selection.flow_id, cs.flow_selection.source) == ("UNRESOLVED", None, "unresolved")
            assert cs.plan is None and cs.tasks == []  # no agent ran on a guess
            assert (cs.pending_hil.type, cs.pending_hil.stage) == ("failure_review", "plan")
            assert any(e.rule == "topic_supported" and not e.passed for e in cs.guardrail_events)
            assert "SELECTED FLOW" not in next(u for r, u in fake.seen if r == "planner") and "FLOWS:" in next(u for r, u in fake.seen if r == "planner")

            cs = await answer(graph, cid, "continue_with_agents", agent_ids=["forecast_agent"])  # a human chooses: the flow follows the choice
            assert (cs.flow_selection.status, cs.flow_selection.flow_id, cs.flow_selection.source) == ("SELECTED", "forecast", "human")
            assert "registry" in cs.flow_selection.reason.lower() and cs.plan.source == "human"
            assert [t.agent_id for t in cs.tasks] == ["forecast_agent"]

    run(scenario())


def test_an_abstaining_planner_on_an_unresolved_case_also_goes_to_a_human(tmp_path, bundle, fake):
    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cs = await start(graph, new_case(bundle, "Something strange happened yesterday, can you please take a look at it soon?"))
            assert cs.flow_selection.status == "UNRESOLVED" and cs.plan is None and cs.tasks == []
            assert any(e.rule == "non_empty" and not e.passed for e in cs.guardrail_events)  # the PLANNER.md abstain rule
            assert (cs.pending_hil.type, cs.pending_hil.stage) == ("failure_review", "plan")

    run(scenario())


def test_when_the_registry_cannot_decide_a_guardrail_passing_planner_flow_is_adopted(tmp_path, bundle, monkeypatch):
    """Only possible if topic support is switched off in guardrails.json: then the planner's flow may stand, and is recorded as such."""
    b = replace(bundle, guardrails={**bundle.guardrails, "plan": {**bundle.guardrails["plan"], "require_topic_support": False}})
    fake = RoutedLLM(b).install(monkeypatch)
    fake.plan_override = {"case_type": "overbooking", "agent_ids": ["overbooking_agent"], "reasoning": "guess"}

    async def scenario():
        async with env(tmp_path, b) as (graph, store):
            cs = await start(graph, new_case(b, "Something strange happened yesterday, can you please take a look at it soon?"))
            assert (cs.flow_selection.status, cs.flow_selection.flow_id, cs.flow_selection.source) == ("SELECTED", "overbooking", "planner")
            assert "Registry could not select a flow" in cs.flow_selection.reason and cs.plan.case_type == "overbooking"

    run(scenario())


# ---------------------------------------------------------------- the planner plans inside the selected flow
def test_the_planner_is_given_only_the_selected_flow(tmp_path, bundle, fake):
    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            await start(graph, new_case(bundle, "Why is my rate only 1000 on 15 August at Hotel Aurora?"))
            planner_msg = next(u for r, u in fake.seen if r == "planner")
            assert "SELECTED FLOW" in planner_msg and "- pricing:" in planner_msg
            assert "- overbooking:" not in planner_msg and "- lrv:" not in planner_msg and "FLOWS:" not in planner_msg

    run(scenario())


def test_a_plan_in_another_flow_is_rejected_and_the_keyword_plan_stays_inside_the_selected_flow(tmp_path, bundle, fake):
    fake.plan_override = {"case_type": "overbooking", "agent_ids": ["overbooking_agent"], "reasoning": "wrong flow for this case"}

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cs = await start(graph, new_case(bundle, "Why is my rate so high compared to competitors this weekend?"))
            failed = {e.rule for e in cs.guardrail_events if e.point == "plan" and not e.passed}
            assert "matches_selected_flow" in failed  # the new rule; the plan was internally consistent (it fits the overbooking flow)
            assert "fits_flow" not in failed
            assert cs.flow_selection.flow_id == "pricing"  # the registry's selection is not overridden by the planner
            assert cs.plan.source == "keyword_fallback" and cs.plan.case_type == "pricing" and cs.plan.agent_ids == ["pricing_agent"]

    run(scenario())


# ---------------------------------------------------------------- persistence across validation, human review, retry and restart
def test_the_selected_flow_survives_validation_failure_retry_and_human_decisions(tmp_path, bundle, fake):
    fake.horizon = "LONG_TERM"  # pricing_agent returns inconclusive evidence, so validation fails and a human is asked

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cid = new_case(bundle, "Why are my rates for next month so low?")
            cs = await start(graph, cid)
            assert (cs.pending_hil.stage, cs.pending_hil.failing_agents) == ("validation", ["pricing_agent"])  # existing HITL behaviour
            before = cs.flow_selection.model_dump(mode="json")
            assert before["flow_id"] == "pricing"
            cs = await answer(graph, cid, "retry:pricing_agent")
            assert len([t for t in cs.tasks if t.agent_id == "pricing_agent"]) == 2 and cs.flow_selection.model_dump(mode="json") == before
            cs = await answer(graph, cid, "continue_with_note", comment="accepted")
            assert cs.pending_hil.type == "final_approval" and cs.flow_selection.model_dump(mode="json") == before
            cs = await answer(graph, cid, "APPROVED")
            assert cs.stage == CaseStatus.COMPLETED and cs.flow_selection.model_dump(mode="json") == before
            assert len(_events(cs, "flow_resolved")) == 1  # the flow was resolved once, not again on retry or resume

    run(scenario())


def test_the_selected_flow_survives_a_restart_and_resume_from_the_checkpoint(tmp_path, bundle, fake):
    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):  # the first 'process': run to the final approval, then stop
            cid = new_case(bundle, "The occupancy forecast for next week looks off compared to last year.")
            cs = await start(graph, cid)
            before = cs.flow_selection.model_dump(mode="json")
            assert before["flow_id"] == "forecast" and cs.pending_hil.type == "final_approval"
        async with env(tmp_path, bundle) as (graph2, store2):  # a new graph over the same checkpoint database and state
            assert state.get(cid).flow_selection.model_dump(mode="json") == before
            cs = await answer(graph2, cid, "APPROVED")
            assert cs.stage == CaseStatus.COMPLETED and cs.flow_selection.model_dump(mode="json") == before

    run(scenario())


def test_an_agent_failure_and_retry_are_visible_in_the_case_record(tmp_path, bundle, fake, monkeypatch):
    from roa.tools import TOOLS

    original = TOOLS["read_lrv"]

    async def boom(env_):
        raise ConnectionError("pricing system unreachable")

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            monkeypatch.setitem(TOOLS, "read_lrv", boom)
            cid = new_case(bundle, "Why is the last room value stuck?")
            cs = await start(graph, cid)
            assert (cs.pending_hil.stage, cs.pending_hil.agent_id) == ("agent", "lrv_agent") and cs.flow_selection.flow_id == "lrv"
            monkeypatch.setitem(TOOLS, "read_lrv", original)
            cs = await answer(graph, cid, "retry")
            assert [t.status.value for t in cs.tasks if t.agent_id == "lrv_agent"] == ["FAILED", "DONE"] and cs.flow_selection.flow_id == "lrv"

    run(scenario())


# ---------------------------------------------------------------- lifecycle telemetry carries the flow
def test_spans_carry_the_selected_flow_and_intent(tmp_path, bundle, fake):
    MEM.clear()

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cid = new_case(bundle, "Why is my rate only 1000 on 15 August at Hotel Aurora?")
            cs = await start(graph, cid)
            await answer(graph, cid, "APPROVED")
            telemetry.flush()
            return state.get(cid)

    cs = run(scenario())
    spans = [s for s in MEM.get_finished_spans() if format(s.context.trace_id, "032x") == cs.trace_id]
    by = {s.name: s.attributes for s in spans}
    fr = by["planner_agent.flow_resolution"]
    assert (fr["roa.flow.id"], fr["roa.flow.status"], fr["roa.flow.source"], fr["roa.intent"]) == ("pricing", "SELECTED", "intent_match", "pricing")
    assert fr["roa.flow.name"] == "Pricing flow" and fr["roa.flow.version"] == "1.0.0" and "registered to flow" in fr["roa.flow.reason"]
    assert by["planner_agent"]["roa.flow.id"] == "pricing" and by["agent.pricing_agent"]["roa.flow.id"] == "pricing"
    assert by["hil.wait"]["roa.flow.id"] == "pricing"
    root = by["case.orchestration"]
    assert (root["roa.flow.id"], root["roa.intent"], root["roa.final_stage"]) == ("pricing", "pricing", "COMPLETED")
    order = [s.name for s in sorted(spans, key=lambda s: s.start_time)]
    assert order.index("planner_agent.understanding") < order.index("planner_agent.flow_resolution") < order.index("planner_agent.plan")
    assert {s.name for s in spans} >= {"intake", "tool.read_pricing_snapshot", "validate.pricing_agent", "reporting_layer"}  # (llm.* spans come from llm.py, which the fake replaces)


def test_llm_calls_are_recorded_per_role_without_prompts_or_keys(tmp_path, bundle, fake):
    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            return await start(graph, new_case(bundle, "Why is my rate only 1000 on 15 August at Hotel Aurora?"))

    cs = run(scenario())
    roles = [c.role for c in cs.llm_usage.detail]
    assert roles == ["understanding", "planner", "validator_judge", "reporter"] and cs.llm_usage.calls == 4
    assert all(set(c.model_dump()) == {"role", "model", "prompt_tokens", "completion_tokens", "latency_ms", "at"} for c in cs.llm_usage.detail)
