"""Deterministic guardrails, driven entirely by harness/guardrails/guardrails.json."""

import re

from roa.harness.loader import HarnessBundle
from roa.models import Evidence, GuardrailEvent

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def check_input(text: str, bundle: HarnessBundle) -> list[GuardrailEvent]:
    g = bundle.guardrails["input"]
    events = [
        GuardrailEvent(
            point="input", rule="length",
            passed=g["min_chars"] <= len(text.strip()) <= g["max_chars"],
            detail=f"{len(text.strip())} chars (allowed {g['min_chars']}-{g['max_chars']})",
        )
    ]
    hit = next((p for p in g["injection_patterns"] if re.search(p, text, re.IGNORECASE)), None)
    events.append(
        GuardrailEvent(point="input", rule="prompt_injection", passed=hit is None,
                       detail=f"matched pattern '{hit}'" if hit else "no injection pattern matched")
    )
    return events


def check_plan(case_type: str, agent_ids: list[str], bundle: HarnessBundle, case_text: str | None = None,
               hint: str | None = None, selected_flow: str | None = None) -> tuple[list[str], list[GuardrailEvent]]:
    """Returns (agents in canonical order, events). The list is only usable if every event passed.
    Pass case_text (and the understanding hint) to also require that each agent is supported by the case; leave it
    None for human-chosen plans, where the human is the authority. Pass selected_flow when the Flow Registry has already
    selected the flow: the plan must then be made inside exactly that flow."""
    g = bundle.guardrails["plan"]
    ev: list[GuardrailEvent] = []
    flow = bundle.flows.get(case_type)
    known = flow is not None and flow.get("enabled", True)
    ev.append(GuardrailEvent(point="plan", rule="known_flow", passed=known, detail=f"case_type={case_type}"))
    if selected_flow is not None:
        ev.append(GuardrailEvent(point="plan", rule="matches_selected_flow", passed=case_type == selected_flow,
                                 detail=(f"plan is in flow '{case_type}', the registry selected '{selected_flow}'"
                                         if case_type != selected_flow else f"plan is in the selected flow '{selected_flow}'")))

    unknown = [a for a in agent_ids if a not in bundle.agents]
    ev.append(GuardrailEvent(point="plan", rule="registered_agents", passed=not (g["require_registered_agents"] and unknown),
                             detail=f"unregistered: {unknown}" if unknown else "all agents registered"))
    disabled = [a for a in agent_ids if a in bundle.agents and not bundle.agents[a].enabled]
    ev.append(GuardrailEvent(point="plan", rule="enabled_agents", passed=not (g["require_enabled_agents"] and disabled),
                             detail=f"disabled: {disabled}" if disabled else "all agents enabled"))
    dupes = sorted({a for a in agent_ids if agent_ids.count(a) > 1})
    ev.append(GuardrailEvent(point="plan", rule="no_duplicates", passed=not (g["forbid_duplicates"] and dupes),
                             detail=f"duplicates: {dupes}" if dupes else "no duplicates"))
    ev.append(GuardrailEvent(point="plan", rule="non_empty", passed=bool(agent_ids), detail=f"{len(agent_ids)} agent(s)"))
    if known:
        outside = [a for a in agent_ids if a not in flow["allowed_agents"]]
        ev.append(GuardrailEvent(point="plan", rule="fits_flow", passed=not (g["must_fit_flow"] and outside),
                                 detail=f"not allowed in flow '{case_type}': {outside}" if outside else f"fits flow '{case_type}'"))
        ev.append(GuardrailEvent(point="plan", rule="max_agents", passed=len(set(agent_ids)) <= flow["max_agents"],
                                 detail=f"{len(set(agent_ids))} of max {flow['max_agents']}"))
    if case_text is not None and g.get("require_topic_support"):
        text = case_text.lower()
        unsupported = []
        for a in agent_ids:
            spec = bundle.agents.get(a)
            by_keyword = bool(spec) and any(k.lower() in text for k in spec.handles)
            hint_flow = bundle.flows.get(hint or "")
            by_hint = bool(hint_flow) and a in hint_flow["allowed_agents"]
            if not (by_keyword or by_hint):
                unsupported.append(a)
        ev.append(GuardrailEvent(point="plan", rule="topic_supported", passed=not unsupported,
                                 detail=(f"no supporting topic in the case text or understanding hint for: {unsupported}"
                                         if unsupported else "every agent is backed by the case text or understanding hint")))
    ordered = bundle.order(list(dict.fromkeys(agent_ids)))
    return ordered, ev


def _numbers(text: str) -> set[str]:
    out = set()
    for tok in _NUM.findall(text):
        tok = tok.replace(",", ".")
        try:
            f = float(tok)
            out.add(str(int(f)) if f == int(f) else str(f))
        except ValueError:
            pass
    return out


def check_output(draft: str, evidence: list[Evidence], bundle: HarnessBundle, extra_allowed_text: str = "") -> list[GuardrailEvent]:
    g = bundle.guardrails["output"]
    ev: list[GuardrailEvent] = []
    if g.get("require_evidence_for_numbers"):
        allowed = _numbers(extra_allowed_text)
        for e in evidence:
            allowed |= _numbers(e.claim)
            allowed |= _numbers(str(e.value))
        stray = sorted(_numbers(draft) - allowed)
        ev.append(GuardrailEvent(point="output", rule="numbers_from_evidence", passed=not stray,
                                 detail=f"numbers not found in evidence: {stray}" if stray else "every number traces to evidence"))
    low = draft.lower()
    bad = [p for p in g.get("forbidden_phrases", []) if p.lower() in low]
    ev.append(GuardrailEvent(point="output", rule="forbidden_phrases", passed=not bad,
                             detail=f"found: {bad}" if bad else "none found"))
    leaks = [p for p in g.get("forbidden_patterns", []) if re.search(p, draft, re.IGNORECASE)]
    ev.append(GuardrailEvent(point="output", rule="no_internal_terms", passed=not leaks,
                             detail=f"internal terms or placeholders found (pattern {leaks[0]})" if leaks else "no internal terms or placeholders"))
    acts = [a for a in g.get("forbidden_actions", []) if a.lower().replace("_", " ") in low or a.lower() in low]
    ev.append(GuardrailEvent(point="output", rule="forbidden_actions", passed=not acts,
                             detail=f"draft claims action: {acts}" if acts else "no side-effect actions claimed"))
    return ev


def failed(events: list[GuardrailEvent]) -> list[GuardrailEvent]:
    return [e for e in events if not e.passed]
