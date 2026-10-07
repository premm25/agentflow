"""The scorer must be trustworthy before its numbers are: test it on hand-built states."""

from evals.scoring import aggregate, gate, judge_metrics, score_case
from roa.context import result_hash
from roa.models import (AgentTaskResult, Case, CaseState, CaseStatus, CheckResult, HILDecision, Plan, TaskRecord,
                        TaskStatus, ToolCallRecord, Verdict)


def _cs(agents=("pricing_agent",), verdict="PASS", stage=CaseStatus.COMPLETED, hil=("report",), draft="All good.",
        bad_check=None):
    cs = CaseState(case=Case(case_id="E1", description="d"), stage=stage,
                   plan=Plan(case_type="pricing", agent_ids=list(agents)) if agents else None,
                   response_draft=draft)
    for a in agents:
        call = ToolCallRecord(agent_id=a, tool="read_pricing_snapshot", result={"rate": 1}, result_hash=result_hash({"rate": 1}))
        cs.tasks.append(TaskRecord(task_id=f"T-{a}", case_id="E1", agent_id=a, status=TaskStatus.DONE, tool_calls=[call],
                                   result=AgentTaskResult(task_id=f"T-{a}", agent_id=a, status=TaskStatus.DONE)))
        checks = [CheckResult(name=bad_check, passed=False)] if bad_check else []
        cs.verdicts.append(Verdict(agent_id=a, task_id=f"T-{a}", attempt=1, status=verdict, checks=checks))
    for s in hil:
        cs.hil_decisions.append(HILDecision(hil_id="h", type="x", stage=s, decision="d"))
    return cs


SPEC = {"id": "c", "expect": {"agents_any_of": [["pricing_agent"]], "hil_path": ["report"],
                              "verdicts": {"pricing_agent": "PASS"}, "final_stage": "COMPLETED"}}


def test_correct_run_passes_every_dimension(bundle):
    assert score_case(SPEC, _cs(), bundle)["passed"]


def test_wrong_plan_is_caught(bundle):
    r = score_case(SPEC, _cs(agents=("pricing_agent", "forecast_agent")), bundle)
    assert not r["plan_ok"] and not r["passed"]


def test_unexpected_human_pause_is_caught(bundle):
    assert not score_case(SPEC, _cs(hil=("validation", "report")), bundle)["hil_ok"]


def test_verdict_expectations(bundle):
    assert score_case(SPEC, _cs(verdict="WARN"), bundle)["verdict_ok"]  # PASS expectation tolerates WARN
    assert not score_case(SPEC, _cs(verdict="FAIL"), bundle)["verdict_ok"]
    fail_spec = {**SPEC, "expect": {**SPEC["expect"], "verdicts": {"pricing_agent": "FAIL"}}}
    assert score_case(fail_spec, _cs(verdict="FAIL"), bundle)["verdict_ok"]
    assert not score_case(fail_spec, _cs(verdict="PASS"), bundle)["verdict_ok"]


def test_ungrounded_evidence_and_unsafe_drafts_are_flagged(bundle):
    assert not score_case(SPEC, _cs(bad_check="numbers_grounded"), bundle)["grounded"]
    assert not score_case(SPEC, _cs(draft="The pricing_agent said your rate is 1200."), bundle)["draft_safe"]


def test_no_plan_expectation(bundle):
    spec = {"id": "v", "expect": {"agents_any_of": [[]], "hil_path": ["plan"], "verdicts": {}, "final_stage": "ABORTED"}}
    ok = _cs(agents=(), stage=CaseStatus.ABORTED, hil=("plan",), draft=None)
    assert score_case(spec, ok, bundle)["passed"]


def test_aggregate_and_judge_metrics_and_gate():
    agg = aggregate([{"passed": True, "plan_ok": True, "hil_ok": True, "verdict_ok": True, "final_ok": True, "grounded": True,
                      "draft_safe": True, "cost": {"llm_calls": 4, "tokens": 100, "llm_ms": 2000}},
                     {"passed": False, "plan_ok": False, "hil_ok": True, "verdict_ok": True, "final_ok": True, "grounded": True,
                      "draft_safe": True, "cost": {"llm_calls": 2, "tokens": 50, "llm_ms": 1000}}])
    assert agg["plan_accuracy"] == 0.5 and agg["avg_llm_calls"] == 3
    jm = judge_metrics([{"label": "FAIL", "verdict": "PASS"}, {"label": "FAIL", "verdict": "FAIL"},
                        {"label": "PASS", "verdict": "FAIL"}, {"label": "PASS", "verdict": "PASS"}])
    assert jm["false_pass_rate"] == 0.5 and jm["false_block_rate"] == 0.5 and jm["accuracy"] == 0.5
    thresholds = {"golden": {"plan_accuracy": 0.85}, "judge": {"max_false_pass_rate": 0.2, "max_false_block_rate": 0.2}}
    fails = gate(agg, jm, thresholds)
    assert any("golden.plan_accuracy" in f for f in fails) and any("false_pass" in f for f in fails)
    assert gate(None, None, thresholds) == []
