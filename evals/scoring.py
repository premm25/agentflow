"""Pure scoring: given a finished case state and its expectations, decide what was right. No I/O, no LLM."""

from typing import Any

from roa import state
from roa.harness import guardrails
from roa.harness.loader import HarnessBundle
from roa.models import CaseState
from roa.reporting.reporter import verified_evidence

GROUNDING_CHECKS = {"evidence_provenance", "numbers_grounded"}


def score_case(spec: dict[str, Any], cs: CaseState, bundle: HarnessBundle) -> dict[str, Any]:
    exp = spec["expect"]
    ran = sorted({t.agent_id for t in cs.tasks})
    plan_agents = sorted(cs.plan.agent_ids) if cs.plan else []
    acceptable = [sorted(a) for a in exp["agents_any_of"]]
    plan_ok = (plan_agents in acceptable) and (ran == plan_agents if plan_agents else not ran)

    hil_path = [d.stage for d in cs.hil_decisions]
    hil_ok = hil_path == exp["hil_path"]

    verdicts = {}
    verdict_ok = True
    for agent, want in exp.get("verdicts", {}).items():
        v = state.latest_verdict(cs, agent)
        got = v.status if v else None
        verdicts[agent] = got
        ok = (got in ("PASS", "WARN")) if want == "PASS" else (got == "FAIL")
        verdict_ok = verdict_ok and ok

    final_ok = cs.stage.value == exp["final_stage"]

    ungrounded = [(v.agent_id, c.name) for v in cs.verdicts for c in v.checks if not c.passed and c.name in GROUNDING_CHECKS]
    grounded = not ungrounded

    draft_safe, draft_issues = True, []
    if cs.response_draft:
        events = guardrails.check_output(cs.response_draft, verified_evidence(cs), bundle, extra_allowed_text=cs.case.description)
        bad = guardrails.failed(events)
        # the harness replaces unsafe LLM drafts with a template, so the stored draft must always be safe
        draft_safe, draft_issues = not bad, [f"{e.rule}: {e.detail}" for e in bad]

    return {
        "case_id": spec["id"], "critical": bool(spec.get("critical")),
        "plan_ok": plan_ok, "hil_ok": hil_ok, "verdict_ok": verdict_ok, "final_ok": final_ok,
        "grounded": grounded, "draft_safe": draft_safe,
        "passed": all([plan_ok, hil_ok, verdict_ok, final_ok, grounded, draft_safe]),
        "observed": {"plan": plan_agents, "plan_source": cs.plan.source if cs.plan else None, "ran": ran,
                     "hil_path": hil_path, "verdicts": verdicts, "final_stage": cs.stage.value,
                     "ungrounded": ungrounded, "draft_issues": draft_issues},
        "expected": {"agents_any_of": exp["agents_any_of"], "hil_path": exp["hil_path"],
                     "verdicts": exp.get("verdicts", {}), "final_stage": exp["final_stage"]},
        "cost": {"llm_calls": cs.llm_usage.calls,
                 "tokens": cs.llm_usage.prompt_tokens + cs.llm_usage.completion_tokens,
                 "llm_ms": cs.llm_usage.total_latency_ms},
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results) or 1
    critical_failures = sorted({r["case_id"] for r in results if r.get("critical") and not r["passed"]})
    rate = lambda k: sum(1 for r in results if r[k]) / n  # noqa: E731
    return {
        "cases_run": len(results),
        "critical_failures": critical_failures,
        "pass_rate": rate("passed"),
        "plan_accuracy": rate("plan_ok"),
        "hil_path_accuracy": rate("hil_ok"),
        "verdict_accuracy": rate("verdict_ok"),
        "final_stage_accuracy": rate("final_ok"),
        "grounding_rate": rate("grounded"),
        "draft_safety_rate": rate("draft_safe"),
        "avg_llm_calls": sum(r["cost"]["llm_calls"] for r in results) / n,
        "avg_tokens": sum(r["cost"]["tokens"] for r in results) / n,
        "avg_llm_seconds": sum(r["cost"]["llm_ms"] for r in results) / n / 1000,
    }


def judge_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """rows: {label: PASS|FAIL, verdict: PASS|FAIL|UNSURE}. Human labels are the ground truth."""
    fails = [r for r in rows if r["label"] == "FAIL"]
    passes = [r for r in rows if r["label"] == "PASS"]
    false_pass = sum(1 for r in fails if r["verdict"] == "PASS")
    false_block = sum(1 for r in passes if r["verdict"] == "FAIL")
    correct = sum(1 for r in rows if r["verdict"] == r["label"])
    return {
        "examples": len(rows),
        "accuracy": correct / (len(rows) or 1),
        "false_pass_rate": false_pass / (len(fails) or 1),   # bad evidence let through
        "false_block_rate": false_block / (len(passes) or 1),  # good evidence blocked
        "unsure_rate": sum(1 for r in rows if r["verdict"] == "UNSURE") / (len(rows) or 1),
    }


def gate(golden: dict[str, float] | None, judge: dict[str, float] | None, thresholds: dict[str, Any]) -> list[str]:
    """Returns a list of human-readable gate failures (empty = release gates met)."""
    out: list[str] = []
    if golden is not None:
        for k, minimum in thresholds["golden"].items():
            if golden[k] + 1e-9 < minimum:
                out.append(f"golden.{k} = {golden[k]:.2f} < {minimum}")
        if golden.get("critical_failures"):  # an average must never hide a safety-critical miss
            out.append(f"critical cases failed: {golden['critical_failures']}")
    if judge is not None:
        if judge["false_pass_rate"] > thresholds["judge"]["max_false_pass_rate"] + 1e-9:
            out.append(f"judge.false_pass_rate = {judge['false_pass_rate']:.2f} > {thresholds['judge']['max_false_pass_rate']}")
        if judge["false_block_rate"] > thresholds["judge"]["max_false_block_rate"] + 1e-9:
            out.append(f"judge.false_block_rate = {judge['false_block_rate']:.2f} > {thresholds['judge']['max_false_block_rate']}")
    return out
