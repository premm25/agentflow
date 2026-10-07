"""Validation layer: checks the correctness of each worker agent's result.

"What is complete" is not in any prompt. It is the agent's own contract in
harness/agents/<id>/AGENT.md (`expected_evidence`) plus harness/validation/validation.json.
Deterministic checks decide first; an LLM judge only covers semantic relevance against that same
declared contract, and cannot invent requirements. Each validator appends to its own log.
"""

import asyncio
import json

from roa import llm, state, telemetry
from roa.harness.guardrails import _numbers
from roa.harness.loader import HarnessBundle
from roa.harness.store import GuardedStore
from roa.models import CaseState, CheckResult, JudgeResult, TaskRecord, TaskStatus, Verdict

PRINCIPAL = "validator"

JUDGE_SCHEMA = {
    "title": "JudgeVerdict",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNSURE"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}


def _deterministic(bundle: HarnessBundle, cs: CaseState, task: TaskRecord) -> list[CheckResult]:
    spec = bundle.agents[task.agent_id]
    sev = bundle.validation["severity"]
    exp = spec.expected_evidence
    res = task.result
    calls = {c.call_id: c for c in task.tool_calls}
    ok_tools = {c.tool for c in task.tool_calls if not c.error}
    checks: list[CheckResult] = []

    def add(name: str, passed: bool, detail: str):
        checks.append(CheckResult(name=name, passed=passed, severity=sev.get(name, "fail"), detail=detail))

    add("status_done", res is not None and res.status == TaskStatus.DONE,
        f"status={res.status.value if res else 'no result'}" + (f" ({res.reason})" if res and res.reason else ""))

    required = exp.get("required_tools", [])
    missing = [t for t in required if t not in ok_tools]
    add("required_tools_called", not missing,
        f"missing required tool calls: {missing}" if missing else f"all required tools called: {required}")

    n = len(res.evidence) if res else 0
    add("min_evidence", n >= exp.get("min_evidence", 1), f"{n} evidence item(s), need {exp.get('min_evidence', 1)}")

    bad_prov: list[str] = []
    bad_nums: list[str] = []
    prop_nums = _numbers(cs.understanding.property_name or "") if cs.understanding else set()
    for e in (res.evidence if res else []):
        call = calls.get(e.source.call_id)
        if call is None or call.tool != e.source.tool or call.result_hash != e.source.result_hash:
            bad_prov.append(f"{e.evidence_id}: not backed by a recorded tool call")
            continue
        for k, v in e.value.items():
            if call.result.get(k) != v:
                bad_prov.append(f"{e.evidence_id}: value '{k}'={v!r} differs from tool result {call.result.get(k)!r}")
        stray = sorted(_numbers(e.claim) - _numbers(json.dumps(call.result)) - prop_nums)
        if stray:
            bad_nums.append(f"{e.evidence_id}: numbers {stray} not in tool output")
    add("evidence_provenance", not bad_prov, "; ".join(bad_prov) or "every evidence item traces to a tool call and matches its output")
    add("numbers_grounded", not bad_nums, "; ".join(bad_nums) or "every number in every claim appears in tool output")

    floor = exp.get("confidence_floor", 0.0)
    e_floor = exp.get("evidence_confidence_floor", 0.0)
    low = [f"{e.evidence_id}={e.confidence}" for e in (res.evidence if res else []) if e.confidence < e_floor]
    conf = res.confidence if res else 0.0
    add("confidence_floor", conf >= floor and not low,
        f"agent confidence {conf} (floor {floor})" + (f"; low-confidence evidence: {low} (floor {e_floor})" if low else ""))

    illegal = sorted({c.tool for c in task.tool_calls} - set(spec.tools_allowed))
    add("allowed_tools_only", not illegal, f"tools outside spec: {illegal}" if illegal else "only allow-listed tools used")
    return checks


async def judge_evidence(bundle: HarnessBundle, case_id: str, case_text: str, spec, evidence_lines: list[str]) -> JudgeResult:
    """Semantic check of returned evidence against the agent's declared contract. Raises llm.LLMError if unavailable.
    Separate from validate_agent so it can be calibrated against labeled examples (evals/)."""
    user = (f"Case:\n{case_text}\n\nAgent: {spec.agent_id}\nDeclared purpose: {spec.purpose}\n"
            f"Declared evidence contract: {json.dumps(spec.expected_evidence)}\n\nEvidence returned:\n"
            + "\n".join(f"- {line}" for line in evidence_lines))
    data, meta = await llm.call_structured(
        case_id, "validator_judge", bundle.model_for("validator_judge"), bundle.judge_prompt, user, JUDGE_SCHEMA,
        max_calls=bundle.guardrails["budgets"]["max_llm_calls_per_case"], timeout=bundle.manifest["llm"]["timeout_s"],
        **bundle.llm_opts("validator_judge"))
    return JudgeResult(verdict=data["verdict"], rationale=data["rationale"], model=meta.model)


async def validate_agent(bundle: HarnessBundle, store: GuardedStore, case_id: str, agent_id: str) -> Verdict:
    cs = state.get(case_id)
    task = state.latest_task(cs, agent_id)
    spec = bundle.agents[agent_id]
    policy = bundle.validation
    log_path = f"cases/{case_id}/validation/{agent_id}.log.jsonl"

    with telemetry.span(f"validate.{agent_id}", "EVALUATOR", **{"roa.case_id": case_id, "roa.agent_id": agent_id,
                                                                  "roa.attempt": task.attempt}) as sp:
        store.log(PRINCIPAL, log_path, "validation_started", task_id=task.task_id, attempt=task.attempt)
        checks = _deterministic(bundle, cs, task)
        failed = [c for c in checks if not c.passed and c.severity == "fail"]
        warned = [c for c in checks if not c.passed and c.severity == "warn"]
        judge: JudgeResult | None = None
        judge_unavailable = False

        if not failed and policy["judge"]["enabled"] and spec.semantic_check:
            ev_lines = [f"{e.claim} (tool={e.source.tool}, confidence={e.confidence})" for e in task.result.evidence]
            try:
                judge = await judge_evidence(bundle, case_id, cs.case.description, spec, ev_lines)
            except llm.LLMError as e:
                judge_unavailable = True  # fallback: deterministic_only, flagged
                store.log(PRINCIPAL, log_path, "judge_unavailable", error=str(e)[:300])

        if failed:
            status = "FAIL"
        elif judge and judge.verdict == "FAIL" and policy["judge"]["fail_blocks"]:
            status = "FAIL"
        elif warned or judge_unavailable or (judge and judge.verdict == "UNSURE"):
            status = "WARN"
        else:
            status = "PASS"

        verdict = Verdict(agent_id=agent_id, task_id=task.task_id, attempt=task.attempt, status=status,
                          checks=checks, judge=judge, judge_unavailable=judge_unavailable)
        state.add_verdict(case_id, verdict)
        sp.set_attribute("roa.verdict", status)
        sp.set_attribute("roa.judge", judge.verdict if judge else ("unavailable" if judge_unavailable else "skipped"))
        sp.set_attribute("roa.checks_failed", json.dumps([c.name for c in checks if not c.passed]))
        store.log(PRINCIPAL, log_path, "verdict", status=status, failed_checks=[c.name for c in checks if not c.passed],
                  judge=judge.model_dump() if judge else None, judge_unavailable=judge_unavailable)
        store.write_json(PRINCIPAL, f"cases/{case_id}/validation/{agent_id}.attempt{task.attempt}.json",
                         verdict.model_dump(mode="json"))
        return verdict


async def run_validation_layer(bundle: HarnessBundle, store: GuardedStore, case_id: str, agent_ids: list[str]) -> list[Verdict]:
    """Validators are independent of each other (each reads one agent's result), so their semantic-judge calls run
    concurrently. The worker agents themselves stay strictly sequential."""
    with telemetry.span("validation_layer", "CHAIN", **{"roa.case_id": case_id, "roa.agents": ",".join(agent_ids)}):
        return list(await asyncio.gather(*(validate_agent(bundle, store, case_id, a) for a in agent_ids)))


def pending_failures(cs: CaseState) -> list[str]:
    """Agents whose latest verdict is FAIL and that a human has not waived."""
    out = []
    for a in (cs.plan.agent_ids if cs.plan else []):
        v = state.latest_verdict(cs, a)
        t = state.latest_task(cs, a)
        if v and v.status == "FAIL" and t and t.status != TaskStatus.SKIPPED and a not in cs.waived_agents:
            out.append(a)
    return out
