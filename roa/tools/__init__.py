"""Tool registry. Placeholder data sources (deterministic per case) standing in for a revenue-management system and a CRM.

Agents never call these directly: they go through AgentContext.call_tool, which enforces the
agent spec's allow-list, records the call, and opens a span.
"""

import asyncio
import random
from typing import Any, Awaitable, Callable

ToolFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
TOOLS: dict[str, ToolFn] = {}


def tool(name: str):
    def deco(fn: ToolFn) -> ToolFn:
        TOOLS[name] = fn
        return fn

    return deco


def _rng(env: dict[str, Any], salt: str = "") -> random.Random:
    return random.Random(f"{env['case_id']}{salt}")


async def _work() -> None:
    await asyncio.sleep(0.05)  # simulated I/O (real system calls would replace these)


def _pricing_snapshot(env: dict[str, Any]) -> dict[str, int]:
    r = _rng(env, ":pricing")
    rate = r.randint(80, 400) * 5
    occ = r.randint(30, 85)
    comp_low = rate + r.randint(200, 800)
    return {"rate": rate, "occupancy_pct": occ, "comp_low": comp_low, "comp_high": comp_low + r.randint(200, 600)}


@tool("read_pricing_snapshot")
async def read_pricing_snapshot(env):
    await _work()
    return {"rate": _pricing_snapshot(env)["rate"]}


@tool("read_occupancy")
async def read_occupancy(env):
    await _work()
    return {"occupancy_pct": _pricing_snapshot(env)["occupancy_pct"]}


@tool("read_competitor_rates")
async def read_competitor_rates(env):
    await _work()
    s = _pricing_snapshot(env)
    return {"comp_low": s["comp_low"], "comp_high": s["comp_high"]}


@tool("read_rate_configuration")
async def read_rate_configuration(env):
    await _work()
    return {"config_changed": False}


@tool("read_booking_status")
async def read_booking_status(env):
    await _work()
    return {"confirmed": _rng(env, ":booking").choice([True, False])}


@tool("read_inventory")
async def read_inventory(env):
    await _work()
    return {"rooms_available": _rng(env, ":inventory").randint(0, 5)}


@tool("read_lrv")
async def read_lrv(env):
    await _work()
    return {"lrv": _rng(env, ":lrv").randint(50, 400) * 5}


@tool("read_lrv_update_log")
async def read_lrv_update_log(env):
    await _work()
    return {"days_since_update": _rng(env, ":lrvlog").randint(1, 10)}


@tool("read_occupancy_forecast")
async def read_occupancy_forecast(env):
    await _work()
    r = _rng(env, ":forecast")
    this_year = r.randint(30, 70)
    last_year = max(5, min(95, this_year + r.randint(-20, 20)))
    return {"forecast_pct": this_year, "prior_year_pct": last_year}


def check_registry(bundle) -> None:
    """Every tool registered in harness/tools/tools.json must have an implementation, and vice versa."""
    from roa.harness.loader import HarnessError

    declared, implemented = set(bundle.tools), set(TOOLS)
    if declared - implemented:
        raise HarnessError(f"tools registered in the harness but not implemented: {sorted(declared - implemented)}")
    if implemented - declared:
        raise HarnessError(f"tools implemented but not registered in the harness: {sorted(implemented - declared)}")
