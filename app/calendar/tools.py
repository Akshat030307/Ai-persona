"""
calendar/tools.py
─────────────────────────────────────────────────────────────────────────────
LangChain Tool definitions wrapping the Google Calendar functions.
These are passed to the LLM as callable tools so it can:
  - check_availability  → returns formatted slot list
  - book_meeting        → creates event, returns confirmation

The LLM decides when to call these based on conversation context.
"""

import json
import logging
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from app.calendar.gcal import get_available_slots, book_meeting, TimeSlot
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Tool Input Schemas ────────────────────────────────────────────────────────

class CheckAvailabilityInput(BaseModel):
    days_ahead: int = Field(
        default=7,
        description="How many days ahead to check for availability. Default is 7.",
        ge=1,
        le=14,
    )


class BookMeetingInput(BaseModel):
    slot_start_iso: str = Field(
        description="ISO 8601 datetime string for the meeting start time (from check_availability output)."
    )
    recruiter_name: str = Field(
        description="Full name of the recruiter or person booking the meeting."
    )
    recruiter_email: str = Field(
        description="Email address of the recruiter to send the calendar invite to."
    )
    meeting_title: Optional[str] = Field(
        default=None,
        description="Optional custom meeting title. Defaults to 'Interview: [candidate] ↔ [recruiter]'.",
    )


# ── Tool Implementations ──────────────────────────────────────────────────────

class CheckAvailabilityTool(BaseTool):
    name: str        = "check_availability"
    description: str = (
        "Check the candidate's real Google Calendar for available meeting slots. "
        "Use this when the recruiter asks about scheduling, availability, or wants to book a call. "
        "Returns a list of available time slots with their ISO datetime strings."
    )
    args_schema: Type[BaseModel] = CheckAvailabilityInput

    def _run(self, days_ahead: int = 7) -> str:
        slots = get_available_slots(days_ahead=days_ahead, max_slots=5)

        if not slots:
            return (
                "No available slots found in the next "
                f"{days_ahead} days during working hours. "
                "The candidate may be fully booked or unavailable."
            )

        result = {
            "available_slots": [s.to_dict() for s in slots],
            "count": len(slots),
            "instructions": (
                "Present 2-3 of these slots to the recruiter. "
                "When they choose one, call book_meeting with the slot_start_iso value."
            )
        }
        return json.dumps(result, indent=2)

    async def _arun(self, days_ahead: int = 7) -> str:
        return self._run(days_ahead)


class BookMeetingTool(BaseTool):
    name: str        = "book_meeting"
    description: str = (
        "Book a confirmed calendar meeting after the recruiter has chosen a time slot. "
        "Requires: the slot ISO datetime (from check_availability), recruiter name, and recruiter email. "
        "This sends a Google Calendar invite to both parties immediately."
    )
    args_schema: Type[BaseModel] = BookMeetingInput

    def _run(
        self,
        slot_start_iso: str,
        recruiter_name: str,
        recruiter_email: str,
        meeting_title: Optional[str] = None,
    ) -> str:
        # Reconstruct slot from ISO string
        from datetime import timedelta
        from zoneinfo import ZoneInfo
        from app.calendar.gcal import SLOT_DURATION_MIN, TZ

        try:
            start = datetime.fromisoformat(slot_start_iso).astimezone(TZ)
        except ValueError as e:
            return json.dumps({"success": False, "error": f"Invalid datetime format: {e}"})

        end  = start + timedelta(minutes=SLOT_DURATION_MIN)
        slot = TimeSlot(start=start, end=end)

        result = book_meeting(
            slot=slot,
            recruiter_name=recruiter_name,
            recruiter_email=recruiter_email,
            meeting_title=meeting_title,
        )

        if result["success"]:
            return json.dumps({
                "success":      True,
                "message":      (
                    f"Meeting confirmed! '{result['title']}' booked for "
                    f"{result['slot_display']}. "
                    f"Calendar invite sent to {recruiter_email}."
                ),
                "meet_link":    result.get("meet_link", ""),
                "event_link":   result.get("event_link", ""),
                "slot_display": result["slot_display"],
            })
        else:
            return json.dumps({
                "success": False,
                "message": f"Booking failed: {result.get('error', 'Unknown error')}. Please try another slot.",
            })

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ── Export tool list ──────────────────────────────────────────────────────────
CALENDAR_TOOLS = [
    CheckAvailabilityTool(),
    BookMeetingTool(),
]
