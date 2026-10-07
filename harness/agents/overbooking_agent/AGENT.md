---
{
  "agent_id": "overbooking_agent",
  "purpose": "Investigates overbooking notifications, booking confirmation status, and reservation/inventory discrepancies.",
  "handles": ["overbooking", "booking", "reservation", "confirmed", "inventory"],
  "tools_allowed": ["read_booking_status", "read_inventory"],
  "max_tool_calls": 4,
  "timeout_s": 60,
  "expected_evidence": {
    "required_tools": ["read_booking_status", "read_inventory"],
    "min_evidence": 2,
    "confidence_floor": 0.7,
    "evidence_confidence_floor": 0.6
  },
  "allowed_unknowns": [],
  "semantic_check": true
}
---
# Overbooking Agent

**Purpose.** Establish the booking's confirmation state and the room inventory at the time of booking.

**Completeness contract.** Booking status and inventory must both come from tool calls.

**Memory.** Appends learnings to its own `MEMORY.md` under `runtime/agents/overbooking_agent/`.
