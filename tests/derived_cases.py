"""Test tooling: validation cases derived from an agent's declared contract.

Reads the *declared structure* of a contract in AGENT.md (min_evidence, confidence floors, required tools, tool allow-list)
and mechanically derives boundary cases from it: one that sits exactly on every boundary (must pass), and cases that step
just past a single boundary. Cases are plain frozen data; `materialize` turns one into a real TaskRecord and `run_case`
runs it through the real `_deterministic` checks. Nothing is generated as code, and no LLM is involved. Derived cases
supplement the hand-written tests in test_validation.py, they do not replace them. This lives under tests/ on purpose: it is
test tooling, not part of the runtime.

Each case names the check it targets (`expect_fail`) and declares every *other* check it is known to trip (`also_fails`),
computed from the declared structure by `_expected` (for example, removing the only required tool leaves no evidence, so
`min_evidence` fails too). `run_case` requires the validator's failures to equal exactly that set: a case that trips an
undeclared check, or fails to trip a declared one, is not ok. `_expected` is an independent restatement of the semantics,
and test_derive.py compares it against the real validator on a grid rather than trusting either.

Assumes a configuration the Config Critic accepts (numeric floors in [0, 1], integer min_evidence >= 1).
"""

from dataclasses import dataclass, replace

from roa.context import result_hash
from roa.harness.loader import HarnessBundle
from roa.models import (AgentTaskResult, Case, CaseState, CaseUnderstanding, Evidence, EvidenceSource, TaskRecord,
                        TaskStatus, ToolCallRecord)
from roa.validation.validator import _deterministic

STEP = 0.01
UNDECLARED_TOOL = "__undeclared_tool__"
_RESULT = {"observed": 1}


@dataclass(frozen=True)
class Scenario:
    """What a worker agent 'did', in the terms the deterministic checks look at."""

    tools_called: tuple[str, ...]  # tools whose output is used as evidence
    evidence_count: int  # evidence items; with no tool called there is nothing to back them, so none exist
    confidence: float
    evidence_confidence: float
    stray_tools: tuple[str, ...] = ()  # tools called but not used as evidence
    status: TaskStatus = TaskStatus.DONE


@dataclass(frozen=True)
class DerivedCase:
    agent_id: str
    name: str
    rationale: str
    scenario: Scenario
    expect_fail: str | None  # the check this case targets; None = every check must pass
    also_fails: frozenset[str] = frozenset()  # other checks this case is known to trip


@dataclass(frozen=True)
class CaseOutcome:
    case: DerivedCase
    failed: frozenset[str]
    ok: bool


def materialize(agent_id: str, sc: Scenario) -> tuple[CaseState, TaskRecord]:
    """Real models, deterministic content: claims contain no numbers, so provenance and grounding hold by construction."""
    cs = CaseState(case=Case(case_id="DERIVED", description="derived scenario"), understanding=CaseUnderstanding(summary="derived"))
    mk = lambda tool: ToolCallRecord(agent_id=agent_id, tool=tool, result=dict(_RESULT), result_hash=result_hash(_RESULT))  # noqa: E731
    calls = [mk(t) for t in sc.tools_called]
    stray = [mk(t) for t in sc.stray_tools]
    evidence = [Evidence(claim="Observed value from a tool call.", value=dict(_RESULT), confidence=sc.evidence_confidence,
                         source=EvidenceSource(tool=c.tool, call_id=c.call_id, result_hash=c.result_hash))
                for c in ([calls[i % len(calls)] for i in range(sc.evidence_count)] if calls else [])]
    res = AgentTaskResult(task_id="T", agent_id=agent_id, status=sc.status, evidence=evidence, confidence=sc.confidence)
    return cs, TaskRecord(task_id="T", case_id="DERIVED", agent_id=agent_id, status=sc.status, result=res,
                          tool_calls=calls + stray)


def _expected(bundle: HarnessBundle, agent_id: str, sc: Scenario) -> set[str]:
    """The deterministic checks a scenario must trip, from the declared contract alone."""
    spec = bundle.agents[agent_id]
    exp = spec.expected_evidence
    called = set(sc.tools_called) | set(sc.stray_tools)
    n = sc.evidence_count if sc.tools_called else 0
    out: set[str] = set()
    if sc.status != TaskStatus.DONE:
        out.add("status_done")
    if any(t not in called for t in exp.get("required_tools", [])):
        out.add("required_tools_called")
    if n < exp.get("min_evidence", 1):
        out.add("min_evidence")
    if sc.confidence < exp.get("confidence_floor", 0.0) or (n > 0 and sc.evidence_confidence < exp.get("evidence_confidence_floor", 0.0)):
        out.add("confidence_floor")
    if called - set(spec.tools_allowed):
        out.add("allowed_tools_only")
    return out


def _case(bundle: HarnessBundle, agent_id: str, name: str, why: str, sc: Scenario, target: str | None) -> DerivedCase:
    expected = _expected(bundle, agent_id, sc)
    if (target is None and expected) or (target is not None and target not in expected):
        raise ValueError(f"{agent_id}/{name}: derivation error, target {target!r} vs expected failures {sorted(expected)}")
    return DerivedCase(agent_id, name, why, sc, target, frozenset(expected - {target}))


def run_case(bundle: HarnessBundle, case: DerivedCase) -> CaseOutcome:
    """Run one derived case through the real deterministic checks. The case is ok only if the failures are exactly the
    declared ones."""
    cs, task = materialize(case.agent_id, case.scenario)
    failed = frozenset(c.name for c in _deterministic(bundle, cs, task) if not c.passed)
    expected = frozenset(case.also_fails | ({case.expect_fail} if case.expect_fail else set()))
    return CaseOutcome(case, failed, failed == expected)


def contract_baseline(bundle: HarnessBundle, agent_id: str) -> Scenario:
    spec = bundle.agents[agent_id]
    exp = spec.expected_evidence
    tools = tuple(exp.get("required_tools", [])) or tuple(spec.tools_allowed[:1])
    return Scenario(tools_called=tools, evidence_count=exp.get("min_evidence", 1), confidence=float(exp.get("confidence_floor", 0.0)),
                    evidence_confidence=float(exp.get("evidence_confidence_floor", 0.0)))


def below(x: float) -> float:
    return round(x - STEP, 4)


def derive_contract_cases(bundle: HarnessBundle, agent_id: str) -> list[DerivedCase]:
    exp = bundle.agents[agent_id].expected_evidence
    base = contract_baseline(bundle, agent_id)
    cases = [_case(bundle, agent_id, "all_boundaries_met", "every contract value sits exactly on its boundary", base, None)]
    if base.evidence_count >= 1:
        cases.append(_case(bundle, agent_id, "min_evidence_below", f"{base.evidence_count - 1} evidence item(s), contract needs {base.evidence_count}",
                           replace(base, evidence_count=base.evidence_count - 1), "min_evidence"))
    for tool in exp.get("required_tools", []):
        cases.append(_case(bundle, agent_id, f"missing_required_tool:{tool}", f"required tool '{tool}' was never called",
                           replace(base, tools_called=tuple(t for t in base.tools_called if t != tool)), "required_tools_called"))
    if base.confidence > 0:
        cases.append(_case(bundle, agent_id, "confidence_below_floor", f"agent confidence just under the {base.confidence} floor",
                           replace(base, confidence=below(base.confidence)), "confidence_floor"))
    if base.evidence_confidence > 0:
        cases.append(_case(bundle, agent_id, "evidence_confidence_below_floor", f"evidence confidence just under the {base.evidence_confidence} floor",
                           replace(base, evidence_confidence=below(base.evidence_confidence)), "confidence_floor"))
    cases.append(_case(bundle, agent_id, "undeclared_tool_called", "a tool outside tools_allowed was called",
                       replace(base, stray_tools=(UNDECLARED_TOOL,)), "allowed_tools_only"))
    cases.append(_case(bundle, agent_id, "agent_reported_failure", "the agent's own status is not DONE",
                       replace(base, status=TaskStatus.FAILED), "status_done"))
    return cases
