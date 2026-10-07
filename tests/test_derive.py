"""Cases derived from the declared contract (tests/derived_cases.py), run through the real deterministic checks."""

from dataclasses import replace

import pytest

from roa.harness import load_harness
from tests.derived_cases import (UNDECLARED_TOOL, DerivedCase, contract_baseline, derive_contract_cases, materialize,
                                   run_case)

AGENTS = ["pricing_agent", "overbooking_agent", "lrv_agent", "forecast_agent"]


def _with_evidence(bundle, agent_id, **changes):
    spec = bundle.agents[agent_id]
    return replace(bundle, agents={**bundle.agents, agent_id: replace(spec, expected_evidence={**spec.expected_evidence, **changes})})


@pytest.mark.parametrize("agent_id", AGENTS)
def test_every_derived_case_behaves_as_derived_for_every_real_agent(bundle, agent_id):
    cases = derive_contract_cases(bundle, agent_id)
    assert cases and cases[0].expect_fail is None
    for case in cases:
        out = run_case(bundle, case)
        assert out.ok, f"{agent_id}/{case.name}: expected {case.expect_fail}, failed={sorted(out.failed)}"


def test_valid_boundary_case_sits_exactly_on_the_contract(bundle):
    base = contract_baseline(bundle, "pricing_agent")
    exp = bundle.agents["pricing_agent"].expected_evidence
    assert (base.evidence_count, base.confidence, base.evidence_confidence) == (3, 0.7, 0.6)
    assert set(base.tools_called) == set(exp["required_tools"])
    assert run_case(bundle, derive_contract_cases(bundle, "pricing_agent")[0]).failed == frozenset()


def test_min_evidence_derives_a_case_on_each_side_of_the_boundary(bundle):
    by_name = {c.name: c for c in derive_contract_cases(bundle, "pricing_agent")}
    assert by_name["all_boundaries_met"].scenario.evidence_count == 3
    assert by_name["min_evidence_below"].scenario.evidence_count == 2
    assert run_case(bundle, by_name["all_boundaries_met"]).failed == frozenset()
    assert "min_evidence" in run_case(bundle, by_name["min_evidence_below"]).failed


def test_confidence_floor_derives_boundary_valid_and_just_below_invalid(bundle):
    by_name = {c.name: c for c in derive_contract_cases(bundle, "pricing_agent")}
    assert by_name["all_boundaries_met"].scenario.confidence == 0.7
    assert by_name["confidence_below_floor"].scenario.confidence == 0.69
    assert "confidence_floor" in run_case(bundle, by_name["confidence_below_floor"]).failed
    assert by_name["evidence_confidence_below_floor"].scenario.evidence_confidence == 0.59
    assert "confidence_floor" in run_case(bundle, by_name["evidence_confidence_below_floor"]).failed


def test_one_missing_required_tool_case_per_required_tool(bundle):
    cases = [c for c in derive_contract_cases(bundle, "pricing_agent") if c.name.startswith("missing_required_tool:")]
    assert [c.name.split(":")[1] for c in cases] == bundle.agents["pricing_agent"].expected_evidence["required_tools"]
    for c in cases:
        assert len(c.scenario.tools_called) == 2
        assert "required_tools_called" in run_case(bundle, c).failed


def test_undeclared_tool_and_failed_status_cases(bundle):
    by_name = {c.name: c for c in derive_contract_cases(bundle, "lrv_agent")}
    assert by_name["undeclared_tool_called"].scenario.stray_tools == (UNDECLARED_TOOL,)
    assert "allowed_tools_only" in run_case(bundle, by_name["undeclared_tool_called"]).failed
    assert "status_done" in run_case(bundle, by_name["agent_reported_failure"]).failed


def test_cases_follow_the_configuration_not_hard_coded_numbers(bundle):
    b = _with_evidence(bundle, "forecast_agent", min_evidence=1, confidence_floor=0.0, evidence_confidence_floor=0.0)
    names = {c.name for c in derive_contract_cases(b, "forecast_agent")}
    assert "confidence_below_floor" not in names and "evidence_confidence_below_floor" not in names  # nothing to step below
    b = _with_evidence(bundle, "pricing_agent", confidence_floor=0.9, min_evidence=3)
    by_name = {c.name: c for c in derive_contract_cases(b, "pricing_agent")}
    assert by_name["confidence_below_floor"].scenario.confidence == 0.89
    assert all(run_case(b, c).ok for c in by_name.values())


def test_derivation_is_deterministic_and_reproducible(bundle):
    assert derive_contract_cases(bundle, "pricing_agent") == derive_contract_cases(bundle, "pricing_agent")


def test_a_derived_case_that_lies_is_reported_not_ok(bundle):
    """The runner must not rubber-stamp: a case claiming a check fails when it does not is flagged."""
    good = derive_contract_cases(bundle, "pricing_agent")[0]
    lie = DerivedCase(good.agent_id, "lie", "claims a failure that is not there", good.scenario, "min_evidence")
    assert not run_case(bundle, lie).ok
    dirty = DerivedCase(good.agent_id, "dirty", "claims clean but confidence is too low",
                        replace(good.scenario, confidence=0.1), None)
    assert not run_case(bundle, dirty).ok


def test_materialized_scenarios_are_real_models(bundle):
    cs, task = materialize("pricing_agent", contract_baseline(bundle, "pricing_agent"))
    assert len(task.result.evidence) == 3 and {c.tool for c in task.tool_calls} == set(
        bundle.agents["pricing_agent"].expected_evidence["required_tools"])
    assert all(e.source.call_id in {c.call_id for c in task.tool_calls} for e in task.result.evidence)


def test_derivation_assumes_a_critic_clean_config(bundle):
    """Derived cases are only meaningful for a configuration the Config Critic accepts; an impossible min_evidence is
    caught by the critic, before any case is derived."""
    from roa.validation.critic import critique_harness

    assert not [f for f in critique_harness(bundle) if f.severity == "error"]
    b = _with_evidence(bundle, "forecast_agent", min_evidence=99)
    assert any(f.code == "impossible_min_evidence" for f in critique_harness(b))


# ---------------------------------------------------------------- exactness: a case says which checks it trips, and the validator must agree
def test_derived_cases_declare_every_check_they_trip(bundle):
    """The forecast agent has a single required tool: removing it leaves nothing to back the evidence, so the case trips
    min_evidence as well. That is declared, not hidden."""
    by_name = {c.name: c for c in derive_contract_cases(bundle, "forecast_agent")}
    missing = by_name["missing_required_tool:read_occupancy_forecast"]
    assert missing.expect_fail == "required_tools_called" and missing.also_fails == {"min_evidence"}
    assert run_case(bundle, missing).failed == {"required_tools_called", "min_evidence"} and run_case(bundle, missing).ok
    pricing = {c.name: c for c in derive_contract_cases(bundle, "pricing_agent")}
    assert all(c.also_fails == frozenset() for n, c in pricing.items())  # pricing has other tools left to back the evidence


def test_a_case_that_trips_an_undeclared_check_is_not_ok(bundle):
    """Regression: 'expect_fail in failed' let a case pass while also tripping unrelated checks."""
    case = {c.name: c for c in derive_contract_cases(bundle, "pricing_agent")}["min_evidence_below"]
    noisy = replace(case, scenario=replace(case.scenario, confidence=0.1))  # also trips confidence_floor
    out = run_case(bundle, noisy)
    assert out.failed == {"min_evidence", "confidence_floor"} and not out.ok


def test_a_case_that_fails_to_trip_a_declared_check_is_not_ok(bundle):
    case = {c.name: c for c in derive_contract_cases(bundle, "pricing_agent")}["min_evidence_below"]
    assert not run_case(bundle, replace(case, also_fails=frozenset({"status_done"}))).ok
    assert not run_case(bundle, replace(case, scenario=contract_baseline(bundle, "pricing_agent"))).ok  # nothing fails at all


def test_the_declared_structure_agrees_with_the_real_validator_on_a_grid(bundle):
    """Differential test: the structural oracle (`_expected`) and the real `_deterministic` must agree on many scenarios."""
    from itertools import product

    from roa.models import TaskStatus
    from tests.derived_cases import Scenario, _expected
    from roa.validation.validator import _deterministic

    for agent_id in bundle.agents:
        spec = bundle.agents[agent_id]
        req = tuple(spec.expected_evidence["required_tools"])
        tool_sets = [req, req[:-1], (), req + ("read_lrv",) if "read_lrv" not in req else req, spec.tools_allowed and tuple(spec.tools_allowed[:1])]
        for tools, count, conf, econf, stray, status in product(tool_sets, (0, 1, 3), (0.5, 0.7, 0.9), (0.5, 0.6, 0.9),
                                                                ((), (UNDECLARED_TOOL,)), (TaskStatus.DONE, TaskStatus.FAILED)):
            sc = Scenario(tuple(tools), count, conf, econf, stray, status)
            cs, task = materialize(agent_id, sc)
            real = {c.name for c in _deterministic(bundle, cs, task) if not c.passed}
            assert real == _expected(bundle, agent_id, sc), (agent_id, sc)


def test_an_inconsistent_derivation_is_an_error_not_a_silent_case(bundle):
    from tests.derived_cases import Scenario, _case

    base = contract_baseline(bundle, "pricing_agent")
    with pytest.raises(ValueError, match="derivation error"):
        _case(bundle, "pricing_agent", "bogus", "claims a failure the contract does not imply", base, "min_evidence")
    with pytest.raises(ValueError, match="derivation error"):
        _case(bundle, "pricing_agent", "bogus", "claims clean but is not", replace(base, confidence=0.1), None)
