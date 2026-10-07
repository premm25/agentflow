"""The Flow Registry (flows.json), intent -> flow resolution, loader validation of the flow metadata, and the plan guardrail."""

import json
import shutil
from dataclasses import replace

import pytest

from roa.config import settings
from roa.harness import HarnessError, guardrails, load_harness
from roa.harness.flows import FlowRegistry
from roa.planner import flow_for_agents

ORDER = ["pricing_agent", "overbooking_agent", "lrv_agent", "forecast_agent"]
REAL_INTENTS = {"pricing": ("pricing",), "overbooking": ("overbooking",), "lrv": ("lrv",), "forecast": ("forecast",), "multi": ()}


def _copy(tmp_path):
    copy = tmp_path / "harness"
    shutil.copytree(settings.harness_dir, copy, copy_function=shutil.copyfile)  # copyfile: do not inherit the read-only lock
    return copy


def _edit_flows(copy, edit):
    p = copy / "flows" / "flows.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    edit(doc["flows"])
    p.write_text(json.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------- the registry over the real flows.json
def test_the_real_registry_exposes_every_existing_flow_with_metadata(bundle):
    reg = bundle.flow_registry()
    specs = {s.flow_id: s for s in reg.all()}
    assert set(specs) == set(bundle.flows) == set(REAL_INTENTS)  # the five existing flows, none added or duplicated
    assert {f: s.intents for f, s in specs.items()} == REAL_INTENTS
    assert all(s.enabled and s.name and s.version and s.description for s in specs.values())
    assert [f for f, s in specs.items() if s.multi_topic] == ["multi"] and reg.multi_flow_id == "multi"
    assert reg.task_order(specs["pricing"]) == ("pricing_agent", "forecast_agent")  # canonical order, not file order
    assert all(a in bundle.agents for s in specs.values() for a in s.allowed_agents)
    assert specs["pricing"].to_dict(reg.task_order(specs["pricing"]))["task_order"] == ["pricing_agent", "forecast_agent"]


def test_an_intent_selects_exactly_one_flow(bundle):
    reg = bundle.flow_registry()
    for intent in ("pricing", "overbooking", "lrv", "forecast"):
        assert reg.for_intent(intent).flow_id == intent
    assert reg.for_intent("multi") is None and reg.for_intent("billing") is None and reg.for_intent(None) is None
    assert reg.for_intent("  LRV ").flow_id == "lrv"  # matching ignores case and spaces


def test_flow_for_agents_keeps_its_existing_behaviour(bundle):
    assert flow_for_agents(bundle, ["forecast_agent"]) == "forecast"
    assert flow_for_agents(bundle, ["pricing_agent", "forecast_agent"]) == "pricing"
    assert flow_for_agents(bundle, ["overbooking_agent", "lrv_agent"]) == "multi"
    assert flow_for_agents(bundle, ["lrv_agent"], "pricing") == "lrv"  # the hint flow does not fit, so the agent's own flow
    assert flow_for_agents(bundle, ["pricing_agent"], "pricing") == "pricing"


def test_keyword_hits_name_the_keywords_and_keyword_match_is_unchanged(bundle):
    text = "The last room value looks stuck and my rate is low"
    hits = bundle.keyword_hits(text)
    assert hits == {"pricing_agent": ["rate"], "lrv_agent": ["last room value"]}
    assert bundle.keyword_match(text) == list(hits) == ["pricing_agent", "lrv_agent"]
    assert bundle.keyword_hits("Something strange happened") == {} and bundle.keyword_match("Something strange happened") == []


# ---------------------------------------------------------------- intent -> flow resolution
@pytest.mark.parametrize("intent,hits,flow,source,reason_has", [
    ("pricing", {"pricing_agent": ["rate"]}, "pricing", "intent_match", "registered to flow 'pricing'"),
    ("pricing", {}, "pricing", "intent_match", "Intent 'pricing'"),  # the intent alone is enough
    ("  Pricing ", {"pricing_agent": ["rate"]}, "pricing", "intent_match", "Intent 'pricing'"),
    ("pricing", {"pricing_agent": ["rate"], "forecast_agent": ["forecast"]}, "pricing", "intent_match", "forecast_agent"),  # forecast may accompany pricing
    ("pricing", {"lrv_agent": ["lrv"]}, "lrv", "keyword_match", "does not cover lrv_agent"),  # the text outweighs a mismatching intent
    ("overbooking", {"overbooking_agent": ["booking"], "lrv_agent": ["lrv"]}, "multi", "multiple_topics", "several topics"),
    (None, {"forecast_agent": ["forecast"]}, "forecast", "keyword_match", "No intent was resolved"),
    ("billing", {"pricing_agent": ["rate"]}, "pricing", "keyword_match", "not registered to any flow"),
    (None, {}, None, "unresolved", "no intent was resolved"),
    ("billing", {}, None, "unresolved", "not registered"),
    ("multi", {}, None, "unresolved", "not registered"),  # 'multi' is a flow, not an intent
])
def test_resolution_rules(bundle, intent, hits, flow, source, reason_has):
    r = bundle.flow_registry().resolve(intent, hits)
    assert (r.flow_id, r.source) == (flow, source) and r.resolved == (flow is not None)
    assert reason_has in r.reason
    assert set(r.matched["keywords"]) == set(hits)  # the evidence behind the decision is returned with it


def test_resolution_is_deterministic(bundle):
    reg = bundle.flow_registry()
    cases = [("pricing", {"pricing_agent": ["rate"], "lrv_agent": ["lrv"]}), (None, {"forecast_agent": ["forecast"]}), ("lrv", {})]
    assert all(len({repr(reg.resolve(i, h)) for _ in range(25)}) == 1 for i, h in cases)
    assert repr(bundle.resolve_flow("pricing", "why is my rate low")) == repr(bundle.resolve_flow("pricing", "why is my rate low"))


def test_a_disabled_flow_is_never_selected(bundle):
    flows = {**bundle.flows, "pricing": {**bundle.flows["pricing"], "enabled": False}}
    reg = FlowRegistry(flows, ORDER)
    assert reg.for_intent("pricing") is None
    r = reg.resolve("pricing", {"pricing_agent": ["rate"]})
    assert r.flow_id != "pricing" and r.source != "intent_match"
    assert reg.resolve("pricing", {}).flow_id is None  # and an intent alone cannot select it


# ---------------------------------------------------------------- loader: flow metadata is validated at startup
def test_an_intent_registered_to_two_flows_is_refused(tmp_path):
    copy = _copy(tmp_path)
    _edit_flows(copy, lambda f: f["overbooking"].update(intents=["pricing"]))
    with pytest.raises(HarnessError, match="registered to both flow 'pricing' and flow 'overbooking'"):
        load_harness(copy, lock=False)


@pytest.mark.parametrize("edit,match", [
    (lambda f: f["pricing"].update(intents="pricing"), "intents must be a list"),
    (lambda f: f["pricing"].update(intents=[""]), "intents must be a list"),
    (lambda f: f["pricing"].update(intents=[5]), "intents must be a list"),
    (lambda f: f["pricing"].update(enabled="yes"), "enabled must be true or false"),
])
def test_malformed_flow_metadata_is_refused(tmp_path, edit, match):
    copy = _copy(tmp_path)
    _edit_flows(copy, edit)
    with pytest.raises(HarnessError, match=match):
        load_harness(copy, lock=False)


def test_a_harness_without_flow_metadata_still_loads_with_defaults(tmp_path):
    """Backward compatibility: flows.json files from before the registry metadata existed keep working."""
    copy = _copy(tmp_path)
    _edit_flows(copy, lambda f: [[d.pop(k, None) for k in ("name", "version", "enabled", "intents", "multi_topic")] for d in f.values()])
    reg = load_harness(copy, lock=False).flow_registry()
    specs = {s.flow_id: s for s in reg.all()}
    assert specs["pricing"].intents == ("pricing",) and specs["pricing"].enabled and specs["pricing"].name == "pricing" and specs["pricing"].version == "0"
    assert specs["multi"].multi_topic and specs["multi"].intents == ()
    assert reg.resolve("lrv", {}).flow_id == "lrv"


def test_the_existing_cross_checks_still_hold(tmp_path):
    copy = _copy(tmp_path)
    _edit_flows(copy, lambda f: f["pricing"]["allowed_agents"].append("ghost_agent"))
    with pytest.raises(HarnessError, match="unknown agent"):
        load_harness(copy, lock=False)


# ---------------------------------------------------------------- plan guardrail
def test_a_plan_must_match_the_flow_the_registry_selected(bundle):
    _, ok = guardrails.check_plan("pricing", ["pricing_agent"], bundle, selected_flow="pricing")
    assert not guardrails.failed(ok) and "matches_selected_flow" in {e.rule for e in ok}
    _, bad = guardrails.check_plan("multi", ["pricing_agent"], bundle, selected_flow="pricing")  # fits multi, but not the selected flow
    assert {e.rule for e in guardrails.failed(bad)} == {"matches_selected_flow"}
    assert "selected 'pricing'" in next(e.detail for e in bad if e.rule == "matches_selected_flow")


def test_without_a_selected_flow_the_guardrail_is_exactly_what_it_was(bundle):
    _, ev = guardrails.check_plan("pricing", ["pricing_agent"], bundle)
    assert "matches_selected_flow" not in {e.rule for e in ev} and not guardrails.failed(ev)


def test_a_disabled_flow_is_not_a_known_flow(bundle):
    b = replace(bundle, flows={**bundle.flows, "pricing": {**bundle.flows["pricing"], "enabled": False}})
    _, ev = guardrails.check_plan("pricing", ["pricing_agent"], b)
    assert "known_flow" in {e.rule for e in guardrails.failed(ev)}
