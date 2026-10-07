"""Agent trajectory: the ordered path one case actually took through the orchestration, rebuilt from its spans.

Planner -> worker agents (each with its ordered tool calls, expected-vs-actual) -> validation verdicts -> human
decisions -> report. Retries show up as extra attempts of the same agent. Pure function over decoded spans, so it is
testable without a server and works for any OTLP source that uses the same span names/attributes.
"""

import json


_STAGE_TEXT = {"report": "response"}  # display name of the HITL stage "report"; the identifier in the data is unchanged


def _j(v, default):
    try:
        return json.loads(v) if isinstance(v, str) else (v if v is not None else default)
    except (ValueError, TypeError):
        return default


def build_trajectory(spans: list[dict]) -> dict:
    """`spans` are span dicts with decoded `attrs` (dict). Returns {phases, path, metrics}."""
    if not spans:
        return {"phases": [], "path": [], "metrics": {}}
    t0 = min(s["start_ns"] for s in spans)
    span_ids = {s["span_id"] for s in spans}
    by_parent: dict = {}
    for s in spans:
        by_parent.setdefault(s["parent_id"], []).append(s)

    def rel(ns):
        return round((ns - t0) / 1e6)

    def dur(s):
        return round((s["end_ns"] - s["start_ns"]) / 1e6)

    def children(s, prefix):
        kids = [c for c in by_parent.get(s["span_id"], []) if c["name"].startswith(prefix)]
        return sorted(kids, key=lambda x: x["start_ns"])

    phases: list[dict] = []
    attempts: dict[str, int] = {}
    for s in sorted(spans, key=lambda x: (x["start_ns"], x["end_ns"])):
        n, a = s["name"], s["attrs"]
        base = {"start_ms": rel(s["start_ns"]), "duration_ms": dur(s), "error": s["status"] == 2}
        if n == "intake":
            phases.append({**base, "type": "intake", "label": "Intake guardrails",
                           "passed": a.get("roa.guardrail.passed", True) is not False})
        elif n == "planner_agent":
            phases.append({**base, "type": "planner", "label": "Planner Agent",
                           "hint": a.get("roa.understanding.hint") or None, "horizon": a.get("roa.understanding.horizon"),
                           "plan_status": a.get("roa.plan.status"), "case_type": a.get("roa.plan.case_type"),
                           "plan_source": a.get("roa.plan.source"), "plan_agents": _j(a.get("roa.plan.agents"), []),
                           "flow_id": a.get("roa.flow.id")})
        elif n == "planner_agent.flow_resolution":
            phases.append({**base, "type": "flow", "label": "Flow selection", "flow_id": a.get("roa.flow.id"), "flow_name": a.get("roa.flow.name"),
                           "version": a.get("roa.flow.version"), "status": a.get("roa.flow.status"), "source": a.get("roa.flow.source"),
                           "reason": a.get("roa.flow.reason"), "intent": a.get("roa.intent") or None, "agents": _j(a.get("roa.flow.agents"), [])})
        elif n.startswith("agent."):
            agent_id = n[len("agent."):]
            attempts[agent_id] = attempts.get(agent_id, 0) + 1
            tools = [{"tool": c["attrs"].get("roa.tool", c["name"][len("tool."):]), "duration_ms": dur(c),
                      "start_ms": rel(c["start_ns"]), "call_id": c["attrs"].get("roa.call_id"), "error": c["status"] == 2}
                     for c in children(s, "tool.")]
            required, called = _j(a.get("roa.required_tools"), []), _j(a.get("roa.tools_called"), [])
            phases.append({**base, "type": "agent", "label": agent_id, "agent_id": agent_id,
                           "attempt": a.get("roa.attempt", attempts[agent_id]), "status": a.get("roa.status"), "flow_id": a.get("roa.flow.id"),
                           "evidence": a.get("roa.evidence_count", 0), "tools": tools, "required_tools": required,
                           "missing_tools": [t for t in required if t not in called],
                           "extra_tools": [t for t in called if t not in required],
                           "tools_correct": all(t in called for t in required)})
        elif n == "validation_layer":
            verdicts = [{"agent_id": c["name"][len("validate."):], "attempt": c["attrs"].get("roa.attempt"),
                         "verdict": c["attrs"].get("roa.verdict"), "failed_checks": _j(c["attrs"].get("roa.checks_failed"), []),
                         "judge": c["attrs"].get("roa.judge"), "duration_ms": dur(c)} for c in children(s, "validate.")]
            phases.append({**base, "type": "validation", "label": "Validation layer", "verdicts": verdicts})
        elif n == "hil.wait":
            phases.append({**base, "type": "hil", "label": f"Human: {_STAGE_TEXT.get(a.get('roa.hil.stage'), a.get('roa.hil.stage'))}", "stage": a.get("roa.hil.stage"),
                           "decision": a.get("roa.hil.decision"), "hil_type": a.get("roa.hil.type"),
                           "agent_id": a.get("roa.hil.agent_id"), "waited_ms": dur(s)})
        elif n == "reporting_layer":
            phases.append({**base, "type": "report", "label": "Response"})
        elif n == "reporting_layer.revision":
            phases.append({**base, "type": "revision", "label": "Response draft revision"})
        elif n == "reporting_layer.report" and s["parent_id"] not in span_ids:
            phases.append({**base, "type": "final_report", "label": "Final report issued"})

    root = next((s for s in spans if s["name"] == "case.orchestration"), None)
    runs = [p for p in phases if p["type"] == "agent"]
    verdicts = [v for p in phases if p["type"] == "validation" for v in p["verdicts"]]
    hil = [p for p in phases if p["type"] == "hil"]
    first_try = [v for v in verdicts if v["attempt"] in (1, None)]

    path = []
    for p in phases:
        t = p["type"]
        if t == "agent":
            path.append({"kind": "agent", "text": f"{p['agent_id']} #{p['attempt']}", "ok": p["status"] == "DONE",
                         "retry": p["attempt"] not in (1, None)})
        elif t == "validation":
            text = ", ".join(f"{v['agent_id'].replace('_agent', '')} {v['verdict']}" for v in p["verdicts"])
            path.append({"kind": "validation", "text": "validate: " + text,
                         "ok": all(v["verdict"] != "FAIL" for v in p["verdicts"])})
        elif t == "hil":
            path.append({"kind": "hil", "text": f"human ({_STAGE_TEXT.get(p['stage'], p['stage'])}): {p['decision']}", "ok": True})
        elif t == "planner":
            path.append({"kind": "planner", "text": "planner" + (" → " + ", ".join(p["plan_agents"]) if p["plan_agents"] else " → no plan"),
                         "ok": p["plan_status"] == "ok"})
        elif t == "flow":
            path.append({"kind": "flow", "text": f"flow: {p['flow_id']}" if p["flow_id"] else "flow: unresolved by registry",
                         "ok": p["status"] == "SELECTED"})
        elif t in ("intake", "report", "revision", "final_report"):
            path.append({"kind": t, "text": p["label"].lower(), "ok": p.get("passed", True)})

    distinct = {p["agent_id"] for p in runs}
    metrics = {
        "agent_runs": len(runs), "distinct_agents": len(distinct), "retries": len(runs) - len(distinct),
        "tool_calls": sum(len(p["tools"]) for p in runs),
        "missing_required_tool_calls": sum(len(p["missing_tools"]) for p in runs),
        "extra_tool_calls": sum(len(p["extra_tools"]) for p in runs),
        "tool_correctness": (sum(1 for p in runs if p["tools_correct"]) / len(runs)) if runs else None,
        "first_try_validation_pass_rate": (sum(1 for v in first_try if v["verdict"] != "FAIL") / len(first_try)) if first_try else None,
        "human_interventions": len(hil), "human_wait_ms": sum(p["waited_ms"] for p in hil),
        "llm_calls": len([s for s in spans if s["name"].startswith("llm.")]),
        "steps": len(phases) + sum(len(p["tools"]) for p in runs),
        "final_stage": root["attrs"].get("roa.final_stage") if root else None,
        "flow_id": next((p["flow_id"] for p in phases if p["type"] in ("flow", "agent") and p.get("flow_id")), None) or (root["attrs"].get("roa.flow.id") if root else None),
    }
    return {"phases": phases, "path": path, "metrics": metrics}
