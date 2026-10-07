"""The execution timeline of one case: trigger -> intent -> flow -> plan -> orchestration -> workers -> validation ->
human -> report -> outcome.

A pure function over the canonical CaseState (the same record the graph writes and the API serves), so what the UI shows is
exactly what the runtime decided and did: the selected flow is read from `state.flow_selection`, never inferred from agent
names. Only components that exist in this runtime appear: there is no QA or governance layer, so none is shown. Durations
come from recorded timestamps (stage transitions, task and tool records, lifecycle events), never from estimates.
"""

from datetime import datetime, timezone
from typing import Any

from roa.harness.loader import HarnessBundle
from roa.models import CaseState, CaseStatus, TaskStatus

FINAL = (CaseStatus.COMPLETED, CaseStatus.FAILED, CaseStatus.ABORTED)
_ROLE_STAGE = {"understanding": "intent", "planner": "plan", "validator_judge": "validation", "reporter": "report"}
_STAGE_TITLES = {"trigger": "Trigger / input", "intake": "Intake guardrails", "intent": "Intent resolution", "flow": "Flow selection",
                 "plan": "Plan", "orchestration": "Orchestration", "workers": "Worker execution", "validation": "Validation",
                 "human": "Human intervention", "report": "Response", "outcome": "Outcome"}
# Display names for identifiers that stay as they are in state and API (a HITL stage "report", the case stage REPORTING).
_STAGE_TEXT = {"report": "response", "REPORTING": "RESPONDING"}


def _txt(v):
    return _STAGE_TEXT.get(v, v)


def _dt(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str):
        try:
            d = datetime.fromisoformat(v)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _ms(a: datetime | None, b: datetime | None) -> int | None:
    return int((b - a).total_seconds() * 1000) if a and b else None


def _segments(cs: CaseState, now: datetime) -> list[tuple[str, datetime, datetime]]:
    """(stage, start, end) from the recorded stage transitions. The last open stage ends now (live) or where it began (final)."""
    marks = [(h["stage"], _dt(h["at"])) for h in cs.history if "stage" in h and _dt(h.get("at"))]
    out = []
    for i, (stage, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else (start if cs.stage in FINAL else now)
        out.append((stage, start, end))
    return out


def build_timeline(cs: CaseState, bundle: HarnessBundle, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    ended = cs.stage in FINAL
    created = _dt(cs.case.created_at)
    segs = _segments(cs, now)
    seg_ms = {}
    for stage, a, b in segs:
        seg_ms[stage] = seg_ms.get(stage, 0) + (_ms(a, b) or 0)
    end_at = segs[-1][1] if ended and segs else None
    events_by_name: dict[str, list[dict]] = {}
    for h in cs.history:
        if "event" in h:
            events_by_name.setdefault(h["event"], []).append(h)
    last_ev = lambda name: (events_by_name.get(name) or [None])[-1]  # noqa: E731

    sel, u, plan = cs.flow_selection, cs.understanding, cs.plan
    not_done = "NOT_REACHED" if ended else "PENDING"
    stages: list[dict[str, Any]] = []

    def stage(key, status, summary, data, started=None, ended_=None, duration=None):
        stages.append({"key": key, "title": _STAGE_TITLES[key], "status": status, "summary": summary, "data": data,
                       "started_at": _iso(started), "ended_at": _iso(ended_), "duration_ms": duration})

    llm_by_stage: dict[str, list[dict]] = {}
    for c in cs.llm_usage.detail:
        llm_by_stage.setdefault(_ROLE_STAGE.get(c.role, c.role), []).append(
            {"role": c.role, "model": c.model, "latency_ms": c.latency_ms, "prompt_tokens": c.prompt_tokens,
             "completion_tokens": c.completion_tokens, "at": _iso(_dt(c.at))})

    # ---- trigger
    stage("trigger", "COMPLETED", f"Case received from {cs.case.source}",
          {"case_id": cs.case.case_id, "source": cs.case.source, "description": cs.case.description,
           "property_name": cs.case.property_name, "received_at": _iso(created)}, created, created, 0)

    # ---- intake guardrails
    intake_ev = [e for e in cs.guardrail_events if e.point in ("harness", "input")]
    blocked = [e for e in intake_ev if not e.passed]
    overridden = any(d.stage == "input" and d.decision == "continue" for d in cs.hil_decisions)
    tampered = cs.stage == CaseStatus.FAILED and "integrity" in (cs.final_reason or "")
    istatus = ("FAILED" if tampered else "OVERRIDDEN" if blocked and overridden else "BLOCKED" if blocked
               else "COMPLETED" if intake_ev else not_done)
    stage("intake", istatus, "; ".join(f"{e.rule}: {'pass' if e.passed else 'FAIL'}" for e in intake_ev) or "not run yet",
          {"events": [{"point": e.point, "rule": e.rule, "passed": e.passed, "detail": e.detail, "at": _iso(_dt(e.at))} for e in intake_ev]},
          _dt(intake_ev[0].at) if intake_ev else None, _dt(intake_ev[-1].at) if intake_ev else None)

    # ---- intent resolution
    ie = last_ev("intent_resolved")
    if u is not None:
        intent = u.case_type_hint
        stage("intent", "COMPLETED", f"Intent: {intent}" if intent else "No intent resolved",
              {"intent": intent, "time_horizon": u.time_horizon.value, "property_name": u.property_name, "summary": u.summary,
               "entities": u.entities, "source": ie.get("source") if ie else None,
               "resolved_by": "understanding LLM" if ie and ie.get("source") == "llm" else "deterministic fallback (LLM unavailable)" if ie else None,
               "llm_calls": llm_by_stage.get("intent", [])},
              None, _dt(ie["at"]) if ie else None, ie.get("duration_ms") if ie else None)
    else:
        stage("intent", "RUNNING" if cs.stage == CaseStatus.PLANNING else not_done, "Resolving the intent of the case", {"llm_calls": []})

    # ---- flow selection (the explicit runtime decision)
    fe = last_ev("flow_resolved")
    reg = bundle.flow_registry()
    if sel is not None:
        spec = reg.get(sel.flow_id)
        considered = [{"flow_id": s.flow_id, "name": s.name, "enabled": s.enabled, "intents": list(s.intents),
                       "selected": s.flow_id == sel.flow_id} for s in reg.all()]
        stage("flow", sel.status,
              f"Flow '{sel.flow_id}' selected ({sel.source.replace('_', ' ')})" if sel.flow_id else "No flow could be selected by the registry",
              {"selection": sel.model_dump(mode="json"), "flow": spec.to_dict(reg.task_order(spec)) if spec else None,
               "considered": considered}, None, _dt(sel.selected_at), fe.get("duration_ms") if fe else None)
    elif plan is not None:  # a case recorded before flow selection existed: say so instead of implying the step is pending
        stage("flow", "NOT_RECORDED", f"No flow selection was recorded for this case (it predates flow resolution); its plan used flow '{plan.case_type}'",
              {"plan_case_type": plan.case_type})
    else:
        stage("flow", "RUNNING" if u is not None and cs.stage == CaseStatus.PLANNING else not_done, "Resolving the flow in the Flow Registry", {})

    # ---- plan
    pe = last_ev("plan_ready") or last_ev("plan_failed")
    plan_guard = [e for e in cs.guardrail_events if e.point == "plan"]
    waiting_plan = cs.pending_hil is not None and cs.pending_hil.stage == "plan"
    if plan is not None:
        stage("plan", "COMPLETED", f"{len(plan.agent_ids)} agent(s): " + " → ".join(plan.agent_ids),
              {"case_type": plan.case_type, "flow_id": sel.flow_id if sel else plan.case_type, "source": plan.source,
               "reasoning": plan.reasoning, "agents": [{"order": i + 1, "agent_id": a} for i, a in enumerate(plan.agent_ids)],
               "execution": "sequential, canonical order", "guardrails": [{"rule": e.rule, "passed": e.passed, "detail": e.detail} for e in plan_guard],
               "llm_calls": llm_by_stage.get("plan", [])}, None, _dt(pe["at"]) if pe else None, pe.get("duration_ms") if pe else None)
    else:
        status = ("WAITING_FOR_HUMAN" if waiting_plan else "FAILED" if (pe and pe["event"] == "plan_failed") else
                  "RUNNING" if sel is not None and cs.stage == CaseStatus.PLANNING else not_done)
        stage("plan", status, "No valid plan: a human chooses the agents" if waiting_plan else "Planning inside the selected flow",
              {"guardrails": [{"rule": e.rule, "passed": e.passed, "detail": e.detail} for e in plan_guard], "reason": pe.get("reason") if pe else None,
               "llm_calls": llm_by_stage.get("plan", [])})

    # ---- orchestration + workers
    agents = list(plan.agent_ids) if plan else []
    latest = {a: next((t for t in reversed(cs.tasks) if t.agent_id == a), None) for a in agents}
    wf = last_ev("workflow_started")
    rows = []
    for i, a in enumerate(agents):
        t = latest[a]
        rows.append({"order": i + 1, "agent_id": a, "status": t.status.value if t else "PENDING",
                     "attempts": len([x for x in cs.tasks if x.agent_id == a]), "started_at": _iso(_dt(t.started_at)) if t else None,
                     "ended_at": _iso(_dt(t.ended_at)) if t else None, "duration_ms": _ms(_dt(t.started_at), _dt(t.ended_at)) if t else None})
    running = next((r["agent_id"] for r in rows if r["status"] == "RUNNING"), None)
    waiting_agent = cs.pending_hil is not None and cs.pending_hil.stage == "agent"
    if not agents:
        ostatus = not_done
    elif running:
        ostatus = "RUNNING"
    elif waiting_agent:
        ostatus = "WAITING_FOR_HUMAN"
    elif any(r["status"] == "FAILED" for r in rows) and ended:
        ostatus = "FAILED"
    elif all(r["status"] in ("DONE", "SKIPPED") for r in rows):
        ostatus = "COMPLETED"
    elif cs.stage == CaseStatus.EXECUTING or rows and any(r["status"] != "PENDING" for r in rows):
        ostatus = "RUNNING"
    else:
        ostatus = not_done
    stage("orchestration", ostatus, f"{sum(1 for r in rows if r['status'] in ('DONE', 'SKIPPED'))} of {len(rows)} task(s) finished" if rows else "Waiting for a plan",
          {"mode": "sequential", "flow_id": sel.flow_id if sel else None, "tasks": rows, "current_agent": running,
           "workflow_started_at": wf["at"] if wf else None}, _dt(wf["at"]) if wf else None, None, seg_ms.get("EXECUTING"))

    workers = []
    for a in agents:
        attempts = []
        for t in [x for x in cs.tasks if x.agent_id == a]:
            r = t.result
            attempts.append({
                "attempt": t.attempt, "task_id": t.task_id, "status": t.status.value, "retry": t.attempt > 1,
                "started_at": _iso(_dt(t.started_at)), "ended_at": _iso(_dt(t.ended_at)), "duration_ms": _ms(_dt(t.started_at), _dt(t.ended_at)),
                "reason": r.reason if r else None, "confidence": r.confidence if r else None, "notes": r.notes if r else None,
                "tool_calls": [{"call_id": c.call_id, "tool": c.tool, "status": "FAILED" if c.error else "COMPLETED", "duration_ms": c.duration_ms,
                                "error": c.error, "at": _iso(_dt(c.at)), "result": c.result, "result_hash": c.result_hash} for c in t.tool_calls],
                "evidence": [{"claim": e.claim, "tool": e.source.tool, "call_id": e.source.call_id, "confidence": e.confidence}
                             for e in (r.evidence if r else [])]})
        workers.append({"agent_id": a, "attempts": attempts})
    wstatus = ("RUNNING" if running else "FAILED" if ended and any(r["status"] == "FAILED" for r in rows) else
               "WAITING_FOR_HUMAN" if waiting_agent else "COMPLETED" if agents and all(r["status"] in ("DONE", "SKIPPED") for r in rows) else not_done if not any(w["attempts"] for w in workers) else "RUNNING")
    tools_n = sum(len(at["tool_calls"]) for w in workers for at in w["attempts"])
    stage("workers", wstatus, f"{tools_n} tool call(s) across {len(agents)} agent(s)", {"agents": workers}, None, None,
          sum(at["duration_ms"] or 0 for w in workers for at in w["attempts"]) if tools_n or agents else None)

    # ---- validation (per-agent verdicts: the runtime validator; its LLM judge shows as llm_calls)
    verdicts = [{"agent_id": v.agent_id, "attempt": v.attempt, "status": v.status, "at": _iso(_dt(v.at)),
                 "failed_checks": [c.name for c in v.checks if not c.passed],
                 "warnings": [c.name for c in v.checks if not c.passed and c.severity == "warn"],
                 "checks": [{"name": c.name, "passed": c.passed, "severity": c.severity, "detail": c.detail} for c in v.checks],
                 "judge": ({"verdict": v.judge.verdict, "rationale": v.judge.rationale, "model": v.judge.model} if v.judge else None),
                 "judge_unavailable": v.judge_unavailable} for v in cs.verdicts]
    final_v = {a: next((v for v in reversed(cs.verdicts) if v.agent_id == a), None) for a in agents}
    vstatus = ("RUNNING" if cs.stage == CaseStatus.VALIDATING else
               "FAILED" if any(v and v.status == "FAIL" and a not in cs.waived_agents for a, v in final_v.items()) and verdicts else
               "COMPLETED" if verdicts else not_done)
    stage("validation", vstatus, ", ".join(f"{a}: {v.status}" for a, v in final_v.items() if v) or "not validated yet",
          {"verdicts": verdicts, "waived_agents": list(cs.waived_agents), "llm_calls": llm_by_stage.get("validation", [])},
          None, None, seg_ms.get("VALIDATING"))

    # ---- human intervention
    decisions = {d.hil_id: d for d in cs.hil_decisions}
    hil_items = []
    for r in cs.hil_requests:
        d = decisions.get(r.hil_id)
        hil_items.append({"hil_id": r.hil_id, "type": r.type, "stage": r.stage, "reason": r.reason, "options": r.options, "agent_id": r.agent_id,
                          "failing_agents": r.failing_agents, "requested_at": _iso(_dt(r.requested_at)),
                          "status": "DECIDED" if d else "PENDING", "decision": d.decision if d else None, "comment": d.comment if d else None,
                          "decided_at": _iso(_dt(d.decided_at)) if d else None, "waited_ms": d.waited_ms if d else None})
    hstatus = "WAITING_FOR_HUMAN" if cs.pending_hil else "COMPLETED" if hil_items else "NOT_REQUIRED" if ended else "PENDING"
    stage("human", hstatus, (f"Waiting: {_txt(cs.pending_hil.stage)} ({cs.pending_hil.type.replace('_', ' ')})" if cs.pending_hil
                             else f"{len(cs.hil_decisions)} decision(s)" if hil_items else "No human intervention"),
          {"items": hil_items, "pending": cs.pending_hil.model_dump(mode="json") if cs.pending_hil else None},
          None, None, sum(d.waited_ms for d in cs.hil_decisions) if cs.hil_decisions else None)

    # ---- reporting
    de = [h for h in cs.history if h.get("event") in ("draft_generated", "draft_revised")]
    rstatus = "COMPLETED" if cs.response_draft and (ended or cs.pending_hil) else "RUNNING" if cs.stage == CaseStatus.REPORTING else not_done
    stage("report", rstatus, "Response draft and report issued" if cs.response_draft else "No response draft yet",
          {"has_draft": bool(cs.response_draft), "draft": cs.response_draft, "draft_source": de[-1].get("source") if de else None,
           "revisions": cs.revision_count, "report_files": sorted(cs.report_paths), "llm_calls": llm_by_stage.get("report", [])},
          None, _dt(de[-1]["at"]) if de else None, seg_ms.get("REPORTING"))

    # ---- outcome
    distinct = {t.agent_id for t in cs.tasks}
    total_ms = _ms(created, end_at or now)
    stage("outcome", cs.stage.value if ended else "IN_PROGRESS", cs.final_reason or ("Running" if not ended else cs.stage.value),
          {"final_stage": cs.stage.value, "reason": cs.final_reason, "total_ms": total_ms, "retries": len(cs.tasks) - len(distinct),
           "human_interventions": len(cs.hil_decisions), "llm_calls": cs.llm_usage.calls,
           "tokens": cs.llm_usage.prompt_tokens + cs.llm_usage.completion_tokens}, None, end_at, total_ms)

    # ---- where the orchestrator is right now
    current = _current(cs, rows, running)

    return {"case_id": cs.case.case_id, "trace_id": cs.trace_id, "status": cs.stage.value, "live": not ended, "current": current,
            "intent": u.case_type_hint if u else None, "flow_id": sel.flow_id if sel else None,
            "flow": sel.model_dump(mode="json") if sel else None, "harness": {"version": cs.harness_version, "hash": cs.harness_hash},
            "started_at": _iso(created), "ended_at": _iso(end_at), "total_ms": total_ms,
            "stage_durations_ms": {"planning": seg_ms.get("PLANNING"), "executing": seg_ms.get("EXECUTING"), "validating": seg_ms.get("VALIDATING"),
                                   "reporting": seg_ms.get("REPORTING"), "waiting_for_human": seg_ms.get("WAITING_FOR_HUMAN")},
            "stages": stages, "events": _events(cs, segs, sel),
            "notes": ["No QA or governance layer exists in this runtime, so none is shown."]}


def _current(cs: CaseState, rows: list[dict], running: str | None) -> dict[str, Any]:
    if cs.stage in FINAL:
        return {"stage": "outcome", "label": f"Finished: {cs.stage.value}", "agent": None}
    if cs.pending_hil:
        return {"stage": "human", "label": f"Waiting for a human: {_txt(cs.pending_hil.stage)} ({cs.pending_hil.type.replace('_', ' ')})",
                "agent": cs.pending_hil.agent_id}
    if cs.stage == CaseStatus.PLANNING:
        step = ("intent" if cs.understanding is None else "flow" if cs.flow_selection is None else "plan")
        return {"stage": step, "label": {"intent": "Resolving the intent", "flow": "Selecting the flow in the registry", "plan": "Planning inside the selected flow"}[step], "agent": None}
    if cs.stage == CaseStatus.EXECUTING:
        nxt = running or next((r["agent_id"] for r in rows if r["status"] == "PENDING"), None)
        return {"stage": "workers", "label": f"Executing {nxt}" if nxt else "Executing worker agents", "agent": nxt}
    if cs.stage == CaseStatus.VALIDATING:
        return {"stage": "validation", "label": "Validating agent results", "agent": None}
    if cs.stage == CaseStatus.REPORTING:
        return {"stage": "report", "label": "Drafting the response and report", "agent": None}
    return {"stage": "intake", "label": "Case received; intake guardrails", "agent": None}


def _events(cs: CaseState, segs: list, sel) -> list[dict[str, Any]]:
    """Flat, time-ordered lifecycle events rebuilt from recorded timestamps. Every event carries the case, and (once a flow is
    selected) the flow and intent, so a case can be reconstructed from this list alone."""
    out: list[dict[str, Any]] = []
    intent = cs.understanding.case_type_hint if cs.understanding else None

    def ev(at, typ, title, stage_key, status=None, **extra):
        d = _dt(at)
        if d is None:
            return
        out.append({"at": _iso(d), "type": typ, "title": title, "stage": stage_key, "status": status, "case_id": cs.case.case_id, **extra})

    ev(cs.case.created_at, "case_received", "Trigger / input received", "trigger", "COMPLETED", source=cs.case.source)
    for h in cs.history:
        if "event" in h:
            e = h["event"]
            title = {"intent_resolved": f"Intent resolved: {h.get('intent') or 'none'} ({h.get('source')})",
                     "flow_resolved": (f"Flow selected: {h.get('flow_id')} ({str(h.get('source')).replace('_', ' ')})" if h.get("flow_id") else "Flow not resolved by the registry"),
                     "plan_ready": f"Plan ready ({h.get('source')}): {', '.join(h.get('agents', []))}", "plan_failed": "Planning produced no valid plan",
                     "planner_fallback": "Planner LLM unavailable or rejected: using the deterministic fallback",
                     "workflow_started": "Workflow started", "draft_generated": f"Response draft generated ({h.get('source')})",
                     "draft_revised": f"Response draft revised ({h.get('source')})"}.get(e, e)
            key = {"intent_resolved": "intent", "flow_resolved": "flow", "plan_ready": "plan", "plan_failed": "plan", "planner_fallback": "plan",
                   "workflow_started": "orchestration", "draft_generated": "report", "draft_revised": "report"}.get(e, "orchestration")
            ev(h["at"], e, title, key, h.get("status") or ("FAILED" if e == "plan_failed" else "COMPLETED"), duration_ms=h.get("duration_ms"))
        elif "stage" in h:
            ev(h["at"], "stage_entered", f"Stage: {_txt(h['stage'])}", "outcome" if h["stage"] in ("COMPLETED", "FAILED", "ABORTED") else "orchestration",
               h["stage"], detail=h.get("reason"))
    for g in cs.guardrail_events:
        ev(g.at, "guardrail", f"Guardrail {g.point}/{g.rule}: {'pass' if g.passed else 'FAIL'}", "intake" if g.point in ("harness", "input") else
           "plan" if g.point == "plan" else "report", "COMPLETED" if g.passed else "FAILED", detail=g.detail)
    for t in cs.tasks:
        ev(t.started_at, "agent_started", f"Agent started: {t.agent_id} (attempt {t.attempt})", "workers", "RUNNING", agent_id=t.agent_id, task_id=t.task_id)
        for c in t.tool_calls:
            ev(c.at, "tool_call", f"Tool {'failed' if c.error else 'completed'}: {c.tool}", "workers", "FAILED" if c.error else "COMPLETED",
               agent_id=t.agent_id, task_id=t.task_id, tool=c.tool, duration_ms=c.duration_ms, detail=c.error)
        if t.ended_at:
            ev(t.ended_at, "agent_finished", f"Agent {t.status.value.lower()}: {t.agent_id} (attempt {t.attempt})", "workers", t.status.value,
               agent_id=t.agent_id, task_id=t.task_id, duration_ms=_ms(_dt(t.started_at), _dt(t.ended_at)),
               detail=(t.result.reason if t.result else None))
    for v in cs.verdicts:
        ev(v.at, "validation_verdict", f"Validation {v.status}: {v.agent_id} (attempt {v.attempt})", "validation", v.status, agent_id=v.agent_id,
           detail=", ".join(c.name for c in v.checks if not c.passed) or None)
    for c in cs.llm_usage.detail:
        ev(c.at, "llm_call", f"LLM call: {c.role} ({c.model})", _ROLE_STAGE.get(c.role, "orchestration"), "COMPLETED", duration_ms=c.latency_ms,
           detail=f"{c.prompt_tokens + c.completion_tokens} tokens")
    for r in cs.hil_requests:
        ev(r.requested_at, "hil_requested", f"Human requested: {_txt(r.stage)} ({r.type.replace('_', ' ')})", "human", "WAITING_FOR_HUMAN", detail=r.reason[:200])
    for d in cs.hil_decisions:
        ev(d.decided_at, "hil_decided", f"Human decided: {d.decision}", "human", "COMPLETED", duration_ms=d.waited_ms, detail=d.comment)
    out.sort(key=lambda e: e["at"])
    sel_at = _iso(_dt(sel.selected_at)) if sel and sel.flow_id else None
    for e in out:  # correlation: once the flow is selected, every later event names it
        if sel_at and e["at"] >= sel_at:
            e["flow_id"], e["intent"] = sel.flow_id, intent
    return out
