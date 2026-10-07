"""Shared test tooling for the multi-flow tests (not a test module).

RoutedLLM is a fake LLM gateway that behaves like the real one where it matters for routing: it infers an intent from the
case text, plans inside the flow named in the SELECTED FLOW block it is sent, records per-call usage with its role, and can be
told to misbehave (propose another flow, be unavailable). Everything else (registry, guardrails, graph, validation, state,
telemetry) is the real runtime.
"""

import re

from roa import llm, state

SELECTED = re.compile(r"SELECTED FLOW[^\n]*\n- (\w+):")


class RoutedLLM:
    def __init__(self, bundle):
        self.bundle = bundle
        self.hint = "auto"  # "auto" = infer from the text, None = no intent, or a fixed string
        self.horizon = "NEAR_TERM"
        self.down: set[str] = set()
        self.plan_override: dict | None = None  # a planner answer to give regardless of the selected flow
        self.judge = {"verdict": "PASS", "rationale": "relevant"}
        self.seen: list[tuple[str, str]] = []  # (role, user message)

    def _infer_hint(self, text: str) -> str | None:
        hits = self.bundle.keyword_hits(text)
        first = next((a for a in self.bundle.canonical_order if a in hits), None)
        return self.bundle.flow_registry().flow_for_agents([first]) if first else None

    async def structured(self, case_id, role, model, system, user, schema, **kw):
        if role in self.down:
            raise llm.LLMError(f"{role} down")
        self.seen.append((role, user))
        state.add_llm_usage(case_id, 10, 5, 1, role=role, model=model)
        text = state.get(case_id).case.description
        if role == "understanding":
            hint = self._infer_hint(text) if self.hint == "auto" else self.hint
            return ({"case_type_hint": hint, "time_horizon": self.horizon, "property_name": None, "entities": {}, "summary": "s"},
                    llm.LLMMeta(model, 10, 5, 1))
        if role == "planner":
            return self._plan(user, text), llm.LLMMeta(model, 10, 5, 1)
        return self.judge, llm.LLMMeta(model, 10, 5, 1)

    def _plan(self, user: str, text: str) -> dict:
        if self.plan_override is not None:
            return self.plan_override
        m = SELECTED.search(user)
        if not m:  # the registry selected nothing: an abstaining planner, as PLANNER.md requires
            return {"case_type": "multi", "agent_ids": [], "reasoning": "no explicit topic"}
        flow = self.bundle.flows[m.group(1)]
        hits = [a for a in self.bundle.keyword_match(text) if a in flow["allowed_agents"]][: flow["max_agents"]]
        return {"case_type": m.group(1), "agent_ids": hits or list(flow["default_agents"]), "reasoning": f"inside {m.group(1)}"}

    async def text(self, case_id, role, model, system, user, **kw):
        if role in self.down:
            raise llm.LLMError(f"{role} down")
        state.add_llm_usage(case_id, 10, 5, 1, role=role, model=model)
        return "Thank you for your case. We reviewed the verified data and will follow up.", llm.LLMMeta(model, 10, 5, 1)

    def install(self, monkeypatch):
        monkeypatch.setattr(llm, "call_structured", self.structured)
        monkeypatch.setattr(llm, "call_text", self.text)
        return self
