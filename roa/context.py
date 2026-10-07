"""AgentContext: the only thing an agent is handed. It is the harness's gateway for tools, logs
and memory, so allow-lists, spans, audit logs and provenance need no per-agent code.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from roa import telemetry
from roa.harness.loader import AgentSpec
from roa.harness.store import GuardedStore
from roa.models import Evidence, EvidenceSource, ToolCallRecord
from roa.tools import TOOLS


class ToolNotAllowed(Exception):
    pass


class ToolBudgetExceeded(Exception):
    pass


def result_hash(result: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:16]


class AgentContext:
    def __init__(self, case_id: str, task_id: str, attempt: int, spec: AgentSpec, store: GuardedStore,
                 tool_env: dict[str, Any], max_tool_calls: int):
        self.case_id = case_id
        self.task_id = task_id
        self.attempt = attempt
        self.spec = spec
        self.agent_id = spec.agent_id
        self.principal = f"agent:{spec.agent_id}"
        self._store = store
        self._env = tool_env
        self._max_calls = min(max_tool_calls, spec.max_tool_calls)
        self.tool_calls: list[ToolCallRecord] = []
        self._reserved = 0  # slots claimed before the await, so parallel calls cannot overshoot the budget

    # ---- paths this agent owns -----------------------------------------------------
    @property
    def _case_dir(self) -> str:
        return f"cases/{self.case_id}/agents/{self.agent_id}"

    @property
    def _memory_path(self) -> str:
        return f"agents/{self.agent_id}/MEMORY.md"

    # ---- tools ---------------------------------------------------------------------
    async def call_tool(self, name: str, **args: Any) -> ToolCallRecord:
        if name not in self.spec.tools_allowed or name not in TOOLS:
            self.log("tool_denied", tool=name)
            raise ToolNotAllowed(f"{self.agent_id} may not call tool '{name}'")
        if self._reserved >= self._max_calls:
            self.log("tool_budget_exceeded", tool=name)
            raise ToolBudgetExceeded(f"{self.agent_id} exceeded {self._max_calls} tool calls")
        self._reserved += 1
        t0 = time.monotonic()
        with telemetry.span(f"tool.{name}", "TOOL", **{"roa.case_id": self.case_id, "roa.agent_id": self.agent_id,
                                                        "roa.tool": name}) as sp:
            try:
                result = await TOOLS[name]({**self._env, **args})
                err = None
            except Exception as e:  # noqa: BLE001 - a failing tool is data, the agent decides how to handle it
                result, err = {}, f"{type(e).__name__}: {e}"
                sp.record_exception(e)
            rec = ToolCallRecord(agent_id=self.agent_id, tool=name, args=args, result=result,
                                 result_hash=result_hash(result), duration_ms=int((time.monotonic() - t0) * 1000),
                                 error=err)
            sp.set_attribute("roa.call_id", rec.call_id)
            sp.set_attribute("roa.result_hash", rec.result_hash)
        self.tool_calls.append(rec)
        self._store.append(self.principal, f"{self._case_dir}/tool_calls.jsonl", rec.model_dump(mode="json"))
        if err:
            raise RuntimeError(f"tool {name} failed: {err}")
        return rec

    async def call_tools(self, *names: str) -> list[ToolCallRecord]:
        """Independent tools of ONE agent run concurrently (results come back in the order asked). Same allow-list,
        budget, spans and audit as call_tool; this is parallelism inside an agent, not between agents."""
        return list(await asyncio.gather(*(self.call_tool(n) for n in names)))

    def claim(self, claim: str, value: dict[str, Any], call: ToolCallRecord, confidence: float) -> Evidence:
        """Evidence is always anchored to a harness-recorded tool call; agents cannot invent sources."""
        return Evidence(claim=claim, value=value, confidence=confidence,
                        source=EvidenceSource(tool=call.tool, call_id=call.call_id, result_hash=call.result_hash))

    # ---- the agent's own logs and memory --------------------------------------------
    def log(self, event: str, **fields: Any) -> None:
        self._store.log(self.principal, f"{self._case_dir}/log.jsonl", event, task_id=self.task_id,
                        attempt=self.attempt, **fields)

    def memory(self, max_chars: int = 2000) -> str:
        return self._store.read(self._memory_path)[-max_chars:]

    def remember(self, text: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._store.append(self.principal, self._memory_path, f"- [{stamp}] {self.case_id}: {text}")
