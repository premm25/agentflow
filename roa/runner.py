"""Harness runner: loads a registered agent by entrypoint and runs it in-process, under a span,
with a timeout, writing its result to its own (write-once) file. Nothing here is agent-specific.
"""

import asyncio
import importlib
import json
from datetime import datetime, timezone

from roa import state, telemetry
from roa.context import AgentContext
from roa.harness.loader import HarnessBundle
from roa.harness.store import GuardedStore
from roa.models import AgentTaskInput, AgentTaskResult, TaskRecord, TaskStatus, new_id


def load_entrypoint(entrypoint: str):
    module_name, _, func = entrypoint.partition(":")
    return getattr(importlib.import_module(module_name), func)


async def run_agent(bundle: HarnessBundle, store: GuardedStore, case_id: str, agent_id: str) -> TaskRecord:
    cs = state.get(case_id)
    spec = bundle.agents[agent_id]
    attempt = len([t for t in cs.tasks if t.agent_id == agent_id]) + 1
    task = TaskRecord(task_id=new_id("TASK"), case_id=case_id, agent_id=agent_id, attempt=attempt)
    state.upsert_task(case_id, task)

    inp = AgentTaskInput(case_id=case_id, task_id=task.task_id, case_text=cs.case.description,
                         understanding=cs.understanding, attempt=attempt)
    tool_env = {"case_id": case_id, "property_name": cs.understanding.property_name,
                "understanding": cs.understanding.model_dump(mode="json")}
    ctx = AgentContext(case_id, task.task_id, attempt, spec, store, tool_env,
                       bundle.guardrails["agent"]["max_tool_calls_per_agent"])

    timeout = min(spec.timeout_s, bundle.guardrails["agent"]["agent_timeout_s"])
    with telemetry.span(f"agent.{agent_id}", "AGENT", **{"roa.case_id": case_id, "roa.agent_id": agent_id,
                                                          "roa.attempt": attempt, "roa.agent_version": spec.version,
                                                          "roa.flow.id": cs.flow_selection.flow_id if cs.flow_selection else None}) as sp:
        try:
            fn = load_entrypoint(spec.entrypoint)
            result: AgentTaskResult = await asyncio.wait_for(fn(ctx, inp), timeout=timeout)
            result = AgentTaskResult.model_validate(result.model_dump())  # contract check
        except asyncio.TimeoutError:
            result = AgentTaskResult(task_id=task.task_id, agent_id=agent_id, status=TaskStatus.FAILED,
                                     reason=f"agent timed out after {timeout}s")
        except Exception as e:  # noqa: BLE001 - any agent-side error becomes a FAILED task for HIL, never a crash
            result = AgentTaskResult(task_id=task.task_id, agent_id=agent_id, status=TaskStatus.FAILED,
                                     reason=f"{type(e).__name__}: {e}")
        sp.set_attribute("roa.status", result.status.value)
        sp.set_attribute("roa.evidence_count", len(result.evidence))
        sp.set_attribute("roa.tool_calls", len(ctx.tool_calls))
        # trajectory: what the contract expected vs what the agent actually did, in order
        sp.set_attribute("roa.required_tools", json.dumps(spec.expected_evidence.get("required_tools", [])))
        sp.set_attribute("roa.allowed_tools", json.dumps(spec.tools_allowed))
        sp.set_attribute("roa.tools_called", json.dumps([c.tool for c in ctx.tool_calls]))

    task.status = result.status
    task.result = result
    task.tool_calls = list(ctx.tool_calls)
    task.ended_at = datetime.now(timezone.utc)
    state.upsert_task(case_id, task)
    ctx.log("finished", status=result.status.value, evidence=len(result.evidence), reason=result.reason)
    store.write_json(ctx.principal, f"cases/{case_id}/agents/{agent_id}/result.attempt{attempt}.json",
                     result.model_dump(mode="json"))
    return task
