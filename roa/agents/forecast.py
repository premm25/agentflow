"""Occupancy / Revenue Forecast Agent."""

from roa.context import AgentContext
from roa.models import AgentTaskInput, AgentTaskResult, TaskStatus


async def run(ctx: AgentContext, inp: AgentTaskInput) -> AgentTaskResult:
    prop = inp.understanding.property_name or "the property"
    ctx.log("started")
    fc = await ctx.call_tool("read_occupancy_forecast")
    this_year, last_year = fc.result["forecast_pct"], fc.result["prior_year_pct"]
    evidence = [ctx.claim(
        f"Occupancy forecast for {prop} for the period in question is {this_year}%, "
        f"compared to {last_year}% for the same period last year.",
        {"forecast_pct": this_year, "prior_year_pct": last_year}, fc, 0.9)]
    ctx.remember(f"forecast {this_year}% vs {last_year}% prior year")
    return AgentTaskResult(task_id=inp.task_id, agent_id=ctx.agent_id, status=TaskStatus.DONE,
                           evidence=evidence, confidence=0.9,
                           notes="Occupancy forecast compared against the same period last year.")
