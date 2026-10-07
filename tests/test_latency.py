"""Latency design: bounded, low-effort LLM calls; independent work concurrent; agents still sequential."""

import asyncio
import json
import time

import httpx
import pytest

from roa import llm
from roa.context import AgentContext, ToolBudgetExceeded
from roa.models import Verdict
from roa.validation import validator


def test_manifest_bounds_every_llm_role(bundle):
    for role in ("understanding", "planner", "validator_judge", "reporter"):
        opts = bundle.llm_opts(role)
        assert opts["reasoning_effort"] == "low", role
        assert 256 <= opts["max_tokens"] <= 2048, role  # a cap, so one call can never run away for a minute


def _client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_chat_sends_effort_and_cap_and_falls_back_when_rejected(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "reasoning_effort" in body and body["model"] == "picky-model":
            return httpx.Response(400, text="unknown param reasoning_effort")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1}})

    async def scenario():
        client = _client_with(handler)
        monkeypatch.setattr(llm, "_client", lambda: client)
        text, _ = await llm._chat([{"role": "user", "content": "x"}], "gpt-oss-20b", 0.1, 5, None, 777, "low")
        assert text == "ok" and seen[-1]["reasoning_effort"] == "low" and seen[-1]["max_tokens"] == 777
        seen.clear()
        text, _ = await llm._chat([{"role": "user", "content": "x"}], "picky-model", 0.1, 5, None, 777, "low")
        assert text == "ok" and len(seen) == 2 and "reasoning_effort" not in seen[-1]  # retried without it
        await client.aclose()

    asyncio.run(scenario())


def test_connection_pool_is_reused_per_event_loop():
    async def scenario():
        a, b = llm._client(), llm._client()
        assert a is b
        await llm.aclose()
        assert llm._client() is not a  # a fresh pool after shutdown
        await llm.aclose()

    asyncio.run(scenario())


def _ctx(bundle, store, agent_id="pricing_agent", max_calls=10):
    return AgentContext("C1", "T1", 1, bundle.agents[agent_id], store, {"case_id": "C1"}, max_calls)


def test_independent_tools_of_one_agent_run_concurrently(bundle, store, monkeypatch):
    from roa import tools

    async def slow(env):
        await asyncio.sleep(0.3)
        return {"v": 1}

    for name in ("read_pricing_snapshot", "read_occupancy", "read_competitor_rates"):
        monkeypatch.setitem(tools.TOOLS, name, slow)

    async def scenario():
        ctx = _ctx(bundle, store)
        t0 = time.monotonic()
        recs = await ctx.call_tools("read_pricing_snapshot", "read_occupancy", "read_competitor_rates")
        elapsed = time.monotonic() - t0
        assert [r.tool for r in recs] == ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates"]  # order kept
        assert elapsed < 0.7, f"three 0.3s tools took {elapsed:.2f}s: not concurrent"
        assert len(store.read_jsonl("cases/C1/agents/pricing_agent/tool_calls.jsonl")) == 3  # still audited

    asyncio.run(scenario())


def test_parallel_tool_calls_cannot_overshoot_the_budget(bundle, store, monkeypatch):
    from roa import tools

    async def slow(env):
        await asyncio.sleep(0.05)
        return {"v": 1}

    for name in ("read_pricing_snapshot", "read_occupancy", "read_competitor_rates"):
        monkeypatch.setitem(tools.TOOLS, name, slow)

    async def scenario():
        ctx = _ctx(bundle, store, max_calls=2)
        results = await asyncio.gather(*(ctx.call_tool(n) for n in ("read_pricing_snapshot", "read_occupancy", "read_competitor_rates")),
                                       return_exceptions=True)
        assert sum(isinstance(r, ToolBudgetExceeded) for r in results) == 1 and len(ctx.tool_calls) == 2

    asyncio.run(scenario())


def test_validators_run_concurrently_across_agents(bundle, store, monkeypatch):
    active, peak = 0, 0

    async def fake_validate(b, s, case_id, agent_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.2)
        active -= 1
        return Verdict(agent_id=agent_id, task_id="t", attempt=1, status="PASS")

    monkeypatch.setattr(validator, "validate_agent", fake_validate)

    async def scenario():
        t0 = time.monotonic()
        out = await validator.run_validation_layer(bundle, store, "C1", ["pricing_agent", "forecast_agent", "lrv_agent"])
        assert [v.agent_id for v in out] == ["pricing_agent", "forecast_agent", "lrv_agent"]  # order preserved
        assert peak == 3 and time.monotonic() - t0 < 0.5

    asyncio.run(scenario())


def test_worker_agents_are_still_sequential():
    """Guard the requirement: the graph runs exactly one worker agent per step (queue.pop(0)), never a gather."""
    import inspect

    import roa.graph as g

    src = inspect.getsource(g)
    assert "queue.pop(0)" in src and "gather" not in src
