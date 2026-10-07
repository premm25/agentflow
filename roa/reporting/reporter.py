"""Final reporting layer. Assembles everything the run produced into report.json / report.md and
drafts the client response from VERIFIED evidence only. Runs after validation; writes only under
runtime/cases/<id>/report/ (the one place a rewrite is allowed, because reports are re-issued at
each revision and at finalization).
"""

from datetime import datetime, timezone

from roa import llm, state, telemetry
from roa.config import settings
from roa.harness import guardrails
from roa.harness.loader import HarnessBundle
from roa.harness.store import GuardedStore
from roa.models import CaseState, Evidence, GuardrailEvent, TaskStatus

PRINCIPAL = "reporter"


def verified_evidence(cs: CaseState) -> list[Evidence]:
    """Evidence from agents whose latest verdict is PASS or WARN. FAIL (even if a human
    continued past it) and skipped agents are never used to write the client response."""
    out: list[Evidence] = []
    for a in (cs.plan.agent_ids if cs.plan else []):
        v, t = state.latest_verdict(cs, a), state.latest_task(cs, a)
        if v and t and t.result and v.status in ("PASS", "WARN") and t.status == TaskStatus.DONE:
            out.extend(t.result.evidence)
    return out


def _caveats(cs: CaseState) -> list[str]:
    cav = list(cs.caveats)
    for a in (cs.plan.agent_ids if cs.plan else []):
        v, t = state.latest_verdict(cs, a), state.latest_task(cs, a)
        if t and t.status == TaskStatus.SKIPPED:
            cav.append(f"{a} was skipped by a human reviewer; its area is not covered.")
        elif v and v.status == "FAIL":
            failing = ", ".join(c.name for c in v.checks if not c.passed) or "semantic judge"
            cav.append(f"{a} failed validation ({failing}); its evidence is unverified and excluded from the draft.")
        elif v and v.status == "WARN":
            cav.append(f"{a} passed with warnings" + (" (semantic judge unavailable)." if v.judge_unavailable else "."))
    return cav


def _client_caveats(bundle: HarnessBundle, cs: CaseState) -> list[str]:
    """Topic-level wording for the client. Internal names, check names and human notes stay in report.md only."""
    out: list[str] = []
    for a in (cs.plan.agent_ids if cs.plan else []):
        v, t = state.latest_verdict(cs, a), state.latest_task(cs, a)
        topic = bundle.agents[a].display_name.replace(" Agent", "").lower()
        if (t and t.status == TaskStatus.SKIPPED) or (v and v.status == "FAIL"):
            line = f"We could not yet verify the {topic} data, so this response does not cover it."
            if line not in out:
                out.append(line)
    return out


def _template_draft(cs: CaseState, ev: list[Evidence], caveats: list[str]) -> str:
    lines = ["Thank you for raising this case. Based on our review of the verified data:"]
    lines += [f"- {e.claim}" for e in ev] or ["- We could not verify enough data to give a confident answer yet."]
    if caveats:
        lines += ["", "Please note:"] + [f"- {c}" for c in caveats]
    lines += ["", "We are happy to look further into any part of this."]
    return "\n".join(lines)


async def _llm_draft(bundle: HarnessBundle, cs: CaseState, ev: list[Evidence], caveats: list[str], extra: str = "") -> tuple[str, str]:
    """Returns (draft, source) where source is 'llm' or 'template'."""
    case_id = cs.case.case_id
    ev_text = "\n".join(f"- {e.claim}" for e in ev) or "(no verified evidence)"
    cav_text = "\n".join(f"- {c}" for c in caveats) or "(none)"
    user = f"Case:\n{cs.case.description}\n\nVerified evidence:\n{ev_text}\n\nCaveats:\n{cav_text}{extra}"
    fb = _template_draft(cs, ev, caveats)
    try:
        draft, _ = await llm.call_text(case_id, "reporter", bundle.model_for("reporter"), bundle.report_prompt, user,
                                       max_calls=bundle.guardrails["budgets"]["max_llm_calls_per_case"],
                                       timeout=bundle.manifest["llm"]["timeout_s"], **bundle.llm_opts("reporter"))
    except llm.LLMError:
        return fb, "template"
    with telemetry.span("guardrail.output", "GUARDRAIL", **{"roa.case_id": case_id}) as sp:
        events = guardrails.check_output(draft, ev, bundle, extra_allowed_text=cs.case.description)
        state.add_guardrail_events(case_id, events)
        bad = guardrails.failed(events)
        sp.set_attribute("roa.guardrail.passed", not bad)
    if bad or not draft:
        state.add_guardrail_events(case_id, [GuardrailEvent(
            point="output", rule="draft_replaced_by_template", passed=True,
            detail="LLM draft violated output guardrails: " + "; ".join(f"{e.rule}: {e.detail}" for e in bad))])
        return fb, "template"
    return draft, "llm"


async def draft_response(bundle: HarnessBundle, store: GuardedStore, case_id: str) -> str:
    cs = state.get(case_id)
    ev, cav = verified_evidence(cs), _client_caveats(bundle, cs)
    with telemetry.span("reporting_layer.draft", "CHAIN", **{"roa.case_id": case_id}) as sp:
        draft, source = await _llm_draft(bundle, cs, ev, cav)
        sp.set_attribute("roa.draft.source", source)
    state.mutate(case_id, lambda s: setattr(s, "response_draft", draft))
    state.record_event(case_id, "draft_generated", source=source)
    store.log(PRINCIPAL, f"cases/{case_id}/report/log.jsonl", "draft_generated", source=source, chars=len(draft))
    return draft


async def revise_response(bundle: HarnessBundle, store: GuardedStore, case_id: str, feedback: str) -> str:
    cs = state.get(case_id)
    ev, cav = verified_evidence(cs), _client_caveats(bundle, cs)
    extra = f"\n\nPrevious draft:\n{cs.response_draft}\n\nReviewer feedback (apply it, stay grounded in the evidence):\n{feedback}"
    with telemetry.span("reporting_layer.revise", "CHAIN", **{"roa.case_id": case_id}) as sp:
        draft, source = await _llm_draft(bundle, cs, ev, cav, extra)
        sp.set_attribute("roa.draft.source", source)
    state.mutate(case_id, lambda s: (setattr(s, "response_draft", draft), setattr(s, "revision_count", s.revision_count + 1)))
    state.record_event(case_id, "draft_revised", source=source)
    store.log(PRINCIPAL, f"cases/{case_id}/report/log.jsonl", "draft_revised", source=source, feedback=feedback[:300])
    return draft


def _report_dict(bundle: HarnessBundle, cs: CaseState) -> dict:
    agents = []
    for a in (cs.plan.agent_ids if cs.plan else []):
        t, v = state.latest_task(cs, a), state.latest_verdict(cs, a)
        agents.append({
            "agent_id": a,
            "attempts": len([x for x in cs.tasks if x.agent_id == a]),
            "status": t.status.value if t else "NOT_RUN",
            "confidence": t.result.confidence if t and t.result else None,
            "notes": t.result.notes if t and t.result else None,
            "reason": t.result.reason if t and t.result else None,
            "evidence": [e.model_dump(mode="json") for e in (t.result.evidence if t and t.result else [])],
            "tool_calls": [{"call_id": c.call_id, "tool": c.tool, "result_hash": c.result_hash,
                            "duration_ms": c.duration_ms} for c in (t.tool_calls if t else [])],
            "verdict": v.model_dump(mode="json") if v else None,
            "waived_by_human": a in cs.waived_agents,
        })
    return {
        "case_id": cs.case.case_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_stage": cs.stage.value,
        "final_reason": cs.final_reason,
        "case": cs.case.model_dump(mode="json"),
        "understanding": cs.understanding.model_dump(mode="json") if cs.understanding else None,
        "flow": cs.flow_selection.model_dump(mode="json") if cs.flow_selection else None,
        "plan": cs.plan.model_dump(mode="json") if cs.plan else None,
        "guardrail_events": [e.model_dump(mode="json") for e in cs.guardrail_events],
        "agents": agents,
        "human_decisions": [d.model_dump(mode="json") for d in cs.hil_decisions],
        "caveats": _caveats(cs),
        "response_draft": cs.response_draft,
        "llm_usage": cs.llm_usage.model_dump(),
        "harness": {"version": cs.harness_version, "hash": cs.harness_hash},
        "observability": {"trace_id": cs.trace_id, "dashboard": f"{settings.dashboard_url}/#trace={cs.trace_id}"},
    }


def _render_md(r: dict) -> str:
    L = [f"# Case report: {r['case_id']}", "",
         f"- Final stage: **{r['final_stage']}**" + (f" ({r['final_reason']})" if r["final_reason"] else ""),
         f"- Generated: {r['generated_at']}",
         f"- Harness: v{r['harness']['version']} (`{r['harness']['hash'][:12]}`)",
         f"- Trace: `{r['observability']['trace_id']}` ({r['observability']['dashboard']})",
         f"- LLM usage: {r['llm_usage']['calls']} calls, {r['llm_usage']['prompt_tokens']} prompt / "
         f"{r['llm_usage']['completion_tokens']} completion tokens", "",
         "## Case", "", r["case"]["description"], ""]
    u = r["understanding"]
    if u:
        L += ["## Understanding", "", f"- Type hint: {u['case_type_hint']}", f"- Horizon: {u['time_horizon']}",
              f"- Property: {u['property_name']}", f"- Summary: {u['summary']}", ""]
    f = r.get("flow")
    if f and f["flow_id"]:
        L += ["## Flow", "", f"- Selected flow: `{f['flow_id']}` ({f['name']} v{f['version']}), source: {f['source']}",
              f"- Intent: {f['intent'] or 'none resolved'}", f"- Why: {f['reason']}",
              f"- Participating agents: {', '.join(f['participating_agents'])}", ""]
    elif f:
        L += ["## Flow", "", f"- No flow was selected by the registry: {f['reason']}", ""]
    p = r["plan"]
    if p:
        L += ["## Plan", "", f"- Flow: `{p['case_type']}` (source: {p['source']})",
              f"- Agents (sequential): {', '.join(p['agent_ids'])}", f"- Reasoning: {p['reasoning']}", ""]
    L += ["## Guardrail events", ""]
    L += [f"- [{'PASS' if e['passed'] else 'FAIL'}] {e['point']}/{e['rule']}: {e['detail']}" for e in r["guardrail_events"]] or ["- none"]
    L += ["", "## Agents and validation", ""]
    for a in r["agents"]:
        v = a["verdict"]
        L += [f"### {a['agent_id']} - {a['status']}" + (f", validation **{v['status']}**" if v else ""),
              f"- attempts: {a['attempts']}, confidence: {a['confidence']}" + (", waived by human" if a["waived_by_human"] else "")]
        L += [f"- evidence: {e['claim']} (via `{e['source']['tool']}`, confidence {e['confidence']})" for e in a["evidence"]]
        if v:
            L += [f"- check [{'ok' if c['passed'] else 'FAILED'}] {c['name']}: {c['detail']}" for c in v["checks"]]
            if v["judge"]:
                L += [f"- judge: {v['judge']['verdict']} - {v['judge']['rationale']}"]
            if v["judge_unavailable"]:
                L += ["- judge: unavailable (deterministic checks only)"]
        L.append("")
    L += ["## Human decisions", ""]
    L += [f"- {d['hil_id']} ({d['type']}/{d['stage']}): **{d['decision']}**" + (f" - {d['comment']}" if d["comment"] else "")
          + f" (waited {d['waited_ms']} ms)" for d in r["human_decisions"]] or ["- none"]
    L += ["", "## Caveats", ""] + ([f"- {c}" for c in r["caveats"]] or ["- none"])
    L += ["", "## Response draft", "", r["response_draft"] or "(not generated)", ""]
    return "\n".join(L)


async def write_report(bundle: HarnessBundle, store: GuardedStore, case_id: str) -> dict[str, str]:
    """(Re)issue report.json and report.md from the current case state."""
    with telemetry.span("reporting_layer.report", "CHAIN", **{"roa.case_id": case_id}):
        cs = state.get(case_id)
        report = _report_dict(bundle, cs)
        j = store.write_json(PRINCIPAL, f"cases/{case_id}/report/report.json", report)
        m = store.write(PRINCIPAL, f"cases/{case_id}/report/report.md", _render_md(report))
        paths = {"json": str(j), "md": str(m)}
        state.mutate(case_id, lambda s: setattr(s, "report_paths", paths))
        store.log(PRINCIPAL, f"cases/{case_id}/report/log.jsonl", "report_written", final_stage=cs.stage.value)
        return paths
