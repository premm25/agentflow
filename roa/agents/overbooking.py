"""Overbooking Agent."""

from roa.context import AgentContext
from roa.models import AgentTaskInput, AgentTaskResult, TaskStatus


async def run(ctx: AgentContext, inp: AgentTaskInput) -> AgentTaskResult:
    prop = inp.understanding.property_name or "the property"
    ctx.log("started")
    booking, inv = await ctx.call_tools("read_booking_status", "read_inventory")
    confirmed = booking.result["confirmed"]
    rooms = inv.result["rooms_available"]
    evidence = [
        ctx.claim(f"Booking for {prop} shows status {'CONFIRMED' if confirmed else 'WAITLISTED'} in the reservation system.",
                  {"confirmed": confirmed}, booking, 0.92),
        ctx.claim(f"Room inventory for the date in question showed {rooms} room(s) remaining at the time of booking.",
                  {"rooms_available": rooms}, inv, 0.88),
    ]
    ctx.remember(f"booking {'confirmed' if confirmed else 'waitlisted'}, {rooms} room(s) left")
    return AgentTaskResult(task_id=inp.task_id, agent_id=ctx.agent_id, status=TaskStatus.DONE,
                           evidence=evidence, confidence=0.9,
                           notes="Booking status and inventory records reviewed.")
