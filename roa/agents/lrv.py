"""Last Room Value (LRV) Agent."""

from roa.context import AgentContext
from roa.models import AgentTaskInput, AgentTaskResult, TaskStatus


async def run(ctx: AgentContext, inp: AgentTaskInput) -> AgentTaskResult:
    prop = inp.understanding.property_name or "the property"
    ctx.log("started")
    lrv, log = await ctx.call_tools("read_lrv", "read_lrv_update_log")
    days = log.result["days_since_update"]
    evidence = [
        ctx.claim(f"Current Last Room Value for {prop} is {lrv.result['lrv']} (local currency).",
                  {"lrv": lrv.result["lrv"]}, lrv, 0.9),
        ctx.claim(f"LRV was last updated by business day-end optimization {days} day(s) ago.",
                  {"days_since_update": days}, log, 0.85),
    ]
    ctx.remember(f"LRV {lrv.result['lrv']}, last updated {days} day(s) ago")
    return AgentTaskResult(task_id=inp.task_id, agent_id=ctx.agent_id, status=TaskStatus.DONE,
                           evidence=evidence, confidence=0.88, notes="LRV value and update history reviewed.")
