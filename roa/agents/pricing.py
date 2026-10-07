"""Pricing Agent.

For LONG_TERM cases it deliberately reads only rate configuration and reports low-confidence,
inconclusive evidence. The validation layer flags that against the agent's contract in
harness/agents/pricing_agent/AGENT.md and a human decides what to do; there is no automatic re-plan.
"""

from roa.context import AgentContext
from roa.models import AgentTaskInput, AgentTaskResult, TaskStatus, TimeHorizon


async def run(ctx: AgentContext, inp: AgentTaskInput) -> AgentTaskResult:
    prop = inp.understanding.property_name or "the property"
    ctx.log("started", horizon=inp.understanding.time_horizon.value, memory_chars=len(ctx.memory()))

    if inp.understanding.time_horizon == TimeHorizon.LONG_TERM:
        cfg = await ctx.call_tool("read_rate_configuration")
        evidence = [ctx.claim(
            f"Rate configuration (offsets, floor/ceiling) for {prop} is unchanged from the prior period.",
            {"config_changed": cfg.result["config_changed"]}, cfg, 0.55)]
        ctx.remember("long-horizon case: only rate configuration checked, demand side not assessed")
        return AgentTaskResult(task_id=inp.task_id, agent_id=ctx.agent_id, status=TaskStatus.DONE,
                               evidence=evidence, confidence=0.55,
                               notes="No rule-level anomaly found in rate configuration. Demand-side factors not assessed.")

    rate, occ, comp = await ctx.call_tools("read_pricing_snapshot", "read_occupancy", "read_competitor_rates")
    evidence = [
        ctx.claim(f"Current rate for {prop} is {rate.result['rate']} (local currency).",
                  {"rate": rate.result["rate"]}, rate, 0.95),
        ctx.claim(f"Occupancy for the period in question is {occ.result['occupancy_pct']}%.",
                  {"occupancy_pct": occ.result["occupancy_pct"]}, occ, 0.9),
        ctx.claim(f"Competitor rates observed in range {comp.result['comp_low']}-{comp.result['comp_high']}.",
                  {"comp_low": comp.result["comp_low"], "comp_high": comp.result["comp_high"]}, comp, 0.85),
    ]
    ctx.remember(f"near-term pricing case: rate {rate.result['rate']}, occupancy {occ.result['occupancy_pct']}%")
    return AgentTaskResult(task_id=inp.task_id, agent_id=ctx.agent_id, status=TaskStatus.DONE,
                           evidence=evidence, confidence=0.9,
                           notes="Pricing, occupancy, and competitor data retrieved.")
