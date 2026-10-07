from roa.context import result_hash
from roa.models import (AgentTaskResult, Case, CaseState, CaseUnderstanding, Evidence, EvidenceSource, TaskRecord,
                        TaskStatus, ToolCallRecord)
from roa.validation.validator import _deterministic


def _cs():
    return CaseState(case=Case(case_id="C1", description="d"), understanding=CaseUnderstanding(summary="s", property_name="Aurora"))


def _call(tool, result):
    return ToolCallRecord(agent_id="pricing_agent", tool=tool, result=result, result_hash=result_hash(result))


def _task(calls, evidence, confidence=0.9, status=TaskStatus.DONE):
    res = AgentTaskResult(task_id="T", agent_id="pricing_agent", status=status, evidence=evidence, confidence=confidence)
    return TaskRecord(task_id="T", case_id="C1", agent_id="pricing_agent", status=status, result=res, tool_calls=calls)


def _ev(call, claim, value, conf=0.9):
    return Evidence(claim=claim, value=value, confidence=conf,
                    source=EvidenceSource(tool=call.tool, call_id=call.call_id, result_hash=call.result_hash))


def _good():
    rate = _call("read_pricing_snapshot", {"rate": 1000})
    occ = _call("read_occupancy", {"occupancy_pct": 60})
    comp = _call("read_competitor_rates", {"comp_low": 1200, "comp_high": 1500})
    ev = [_ev(rate, "Current rate is 1000.", {"rate": 1000}), _ev(occ, "Occupancy is 60%.", {"occupancy_pct": 60}),
          _ev(comp, "Competitors at 1200-1500.", {"comp_low": 1200, "comp_high": 1500})]
    return [rate, occ, comp], ev


def _failed(checks):
    return {c.name for c in checks if not c.passed}


def test_valid_result_passes_every_deterministic_check(bundle):
    calls, ev = _good()
    assert _failed(_deterministic(bundle, _cs(), _task(calls, ev))) == set()


def test_missing_required_tool_and_evidence_fails(bundle):
    calls, ev = _good()
    failed = _failed(_deterministic(bundle, _cs(), _task(calls[:1], ev[:1])))
    assert {"required_tools_called", "min_evidence"} <= failed


def test_fabricated_evidence_without_a_tool_call_fails_provenance(bundle):
    calls, ev = _good()
    fake = Evidence(claim="Current rate is 1000.", value={"rate": 1000}, confidence=0.9,
                    source=EvidenceSource(tool="read_pricing_snapshot", call_id="CALL-fake", result_hash="x"))
    failed = _failed(_deterministic(bundle, _cs(), _task(calls, [fake] + ev[1:])))
    assert "evidence_provenance" in failed


def test_value_that_differs_from_tool_output_fails(bundle):
    calls, ev = _good()
    ev[0] = _ev(calls[0], "Current rate is 1000.", {"rate": 9999})
    assert "evidence_provenance" in _failed(_deterministic(bundle, _cs(), _task(calls, ev)))


def test_number_in_claim_not_in_tool_output_fails(bundle):
    calls, ev = _good()
    ev[0] = _ev(calls[0], "Current rate is 1200.", {"rate": 1000})
    assert "numbers_grounded" in _failed(_deterministic(bundle, _cs(), _task(calls, ev)))


def test_low_confidence_fails_floor(bundle):
    calls, ev = _good()
    assert "confidence_floor" in _failed(_deterministic(bundle, _cs(), _task(calls, ev, confidence=0.55)))


def test_failed_status_fails(bundle):
    calls, ev = _good()
    assert "status_done" in _failed(_deterministic(bundle, _cs(), _task(calls, ev, status=TaskStatus.FAILED)))


def test_tool_outside_spec_is_flagged(bundle):
    calls, ev = _good()
    calls.append(_call("read_lrv", {"lrv": 500}))
    assert "allowed_tools_only" in _failed(_deterministic(bundle, _cs(), _task(calls, ev)))
