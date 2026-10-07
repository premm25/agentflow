"""Flow Registry: typed, read-only access to harness/flows/flows.json and the deterministic intent -> flow resolution.

There is one registry: the flows file. This module is the API over it (the bundle builds it from the same dicts), so the
planner, the guardrails, the API and the UI all read the same definitions. Resolution is pure and deterministic: given the
resolved intent and which agents' keywords appear in the case text, it returns the registered flow to use, and *why*. It
never calls an LLM. The LLM planner then plans inside the flow it is given.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlowSpec:
    flow_id: str
    name: str
    version: str
    description: str
    enabled: bool
    intents: tuple[str, ...]
    allowed_agents: tuple[str, ...]
    default_agents: tuple[str, ...]
    max_agents: int
    multi_topic: bool

    def to_dict(self, task_order: tuple[str, ...] = ()) -> dict[str, Any]:
        return {"flow_id": self.flow_id, "name": self.name, "version": self.version, "description": self.description,
                "enabled": self.enabled, "intents": list(self.intents), "allowed_agents": list(self.allowed_agents),
                "default_agents": list(self.default_agents), "max_agents": self.max_agents, "multi_topic": self.multi_topic,
                "task_order": list(task_order or self.allowed_agents)}


@dataclass(frozen=True)
class FlowResolution:
    flow_id: str | None
    source: str  # intent_match | keyword_match | multiple_topics | unresolved
    reason: str
    intent: str | None
    matched: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.flow_id is not None


class FlowRegistry:
    def __init__(self, flows: dict[str, dict[str, Any]], canonical_order: list[str]):
        self._order = list(canonical_order)
        self._specs: dict[str, FlowSpec] = {fid: self._spec(fid, f) for fid, f in flows.items()}
        self.multi_flow_id = next((s.flow_id for s in self._specs.values() if s.multi_topic), "multi")

    @staticmethod
    def _spec(flow_id: str, f: dict[str, Any]) -> FlowSpec:
        multi = bool(f.get("multi_topic", flow_id == "multi"))
        return FlowSpec(
            flow_id=flow_id, name=f.get("name", flow_id), version=str(f.get("version", "0")), description=f.get("description", ""),
            enabled=bool(f.get("enabled", True)), intents=tuple(f["intents"]) if "intents" in f else (() if multi else (flow_id,)),
            allowed_agents=tuple(f["allowed_agents"]), default_agents=tuple(f.get("default_agents", [])),
            max_agents=int(f["max_agents"]), multi_topic=multi)

    # ---- lookup -------------------------------------------------------------------
    def get(self, flow_id: str | None) -> FlowSpec | None:
        return self._specs.get(flow_id or "")

    def all(self) -> list[FlowSpec]:
        return list(self._specs.values())

    def for_intent(self, intent: str | None) -> FlowSpec | None:
        intent = (intent or "").strip().lower()
        return next((s for s in self._specs.values() if s.enabled and intent and intent in s.intents), None)

    def task_order(self, spec: FlowSpec) -> tuple[str, ...]:
        """Agents of the flow in the order the orchestrator runs them (the global canonical order)."""
        rank = {a: i for i, a in enumerate(self._order)}
        return tuple(sorted(spec.allowed_agents, key=lambda a: rank.get(a, len(rank))))

    def menu_line(self, flow_id: str) -> str:
        s = self._specs[flow_id]
        return f"- {s.flow_id}: {s.description} (allowed agents: {', '.join(s.allowed_agents)})"

    # ---- selection ----------------------------------------------------------------
    def flow_for_agents(self, agent_ids: list[str], hint: str | None = None) -> str:
        """The flow a deterministic (non-LLM) agent selection belongs to."""
        ok = lambda s: s.enabled and all(a in s.allowed_agents for a in agent_ids)  # noqa: E731
        if hint in self._specs and ok(self._specs[hint]):
            return hint
        if len(agent_ids) == 1:
            for s in self._specs.values():
                if s.enabled and agent_ids[0] in s.default_agents:
                    return s.flow_id
        for s in self._specs.values():
            if ok(s) and len(agent_ids) <= s.max_agents:
                return s.flow_id
        return self.multi_flow_id

    def resolve(self, intent: str | None, hits: dict[str, list[str]]) -> FlowResolution:
        """Intent + keyword evidence -> a registered flow and the reason. `hits` maps agent id -> the agent's `handles`
        keywords found in the case text. Deterministic: the same input always selects the same flow."""
        intent = (intent or "").strip().lower() or None
        intent_flow = self.for_intent(intent)
        topics = [a for a in self._order if a in hits] + [a for a in hits if a not in self._order]
        matched = {"intent": intent, "intent_flow": intent_flow.flow_id if intent_flow else None,
                   "keywords": {a: list(hits[a]) for a in topics}}

        if intent_flow and all(a in intent_flow.allowed_agents for a in topics):
            also = f"; case text also mentions {', '.join(topics)}" if topics else ""
            return FlowResolution(intent_flow.flow_id, "intent_match",
                                  f"Intent '{intent}' is registered to flow '{intent_flow.flow_id}'{also}", intent, matched)
        if topics:
            fid = self.flow_for_agents(topics, intent)
            spec = self.get(fid)
            if spec is not None:
                if intent_flow:
                    outside = [a for a in topics if a not in intent_flow.allowed_agents]
                    lead = f"Intent '{intent}' maps to flow '{intent_flow.flow_id}', which does not cover {', '.join(outside)}; "
                elif intent:
                    lead = f"Intent '{intent}' is not registered to any flow; "
                else:
                    lead = "No intent was resolved; "
                if spec.multi_topic and len(topics) > 1:
                    return FlowResolution(fid, "multiple_topics", lead + f"the case text raises several topics ({', '.join(topics)}) "
                                          f"that no single flow covers, so the multi-topic flow '{fid}' is used", intent, matched)
                return FlowResolution(fid, "keyword_match", lead + f"the case text matches {', '.join(topics)}, so flow '{fid}' is used",
                                      intent, matched)
        why = "no intent was resolved" if not intent else f"intent '{intent}' is not registered to any enabled flow"
        return FlowResolution(None, "unresolved", f"Registry could not select a flow: {why} and no agent keywords matched the case text",
                              intent, matched)
