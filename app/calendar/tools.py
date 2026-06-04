"""
calendar/tools.py
LangChain Tool definitions wrapping Google Calendar functions.
Includes email cleaning to handle voice-transcribed emails like "a k s h a t at gmail dot com"
"""

import json
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Type
from zoneinfo import ZoneInfo

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from app.calendar.gcal import get_available_slots, book_meeting, TimeSlot, SLOT_DURATION_MIN, TZ

logger = logging.getLogger(__name__)


# ── Email Cleaner ─────────────────────────────────────────────────────────────
def _clean_email(raw: str) -> str:
    """
    Clean up voice-transcribed emails.
    Handles patterns like:
      - "a k s h a t at gmail dot com" → "akshat@gmail.com"
      - "akshat at the rate of gmail dot com" → "akshat@gmail.com"
      - "akshat2006 at gmail.com" → "akshat2006@gmail.com"
    """
    email = raw.lower().strip()

    # Remove spaces between individual letters (spelled out)
    # e.g. "a k s h a t" → "akshat"
    email = re.sub(r'(?<=[a-z0-9]) (?=[a-z0-9])', '', email)

    # Replace "at the rate of", "at the rate", "at" with @
    email = re.sub(r'\s*at the rate of\s*', '@', email)
    email = re.sub(r'\s*at the rate\s*',    '@', email)
    email = re.sub(r'\s+at\s+',             '@', email)

    # Replace "dot" with .
    email = re.sub(r'\s*dot\s*', '.', email)

    # Remove any remaining spaces
    email = email.replace(' ', '')

    logger.info(f"Cleaned email: '{raw}' → '{email}'")
    return email


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email))


# ── Tool Input Schemas ────────────────────────────────────────────────────────
class CheckAvailabilityInput(BaseModel):
    days_ahead: int = Field(
        default=7,
        description="How many days ahead to check for availability. Default is 7.",
        ge=1, le=14,
    )


class BookMeetingInput(BaseModel):
    slot_start_iso: str = Field(
        description="ISO 8601 datetime string for the meeting start time (from check_availability output)."
    )
    recruiter_name: str = Field(
        description="Full name of the recruiter or person booking the meeting."
    )
    recruiter_email: str = Field(
        description="Email address of the recruiter. May be transcribed from voice, e.g. 'name at gmail dot com'."
    )
    meeting_title: Optional[str] = Field(
        default=None,
        description="Optional custom meeting title.",
    )


# ── Check Availability Tool ───────────────────────────────────────────────────
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
            return json.dumps({
                "available_slots": [],
                "message": f"No available slots found in the next {days_ahead} days."
            })
        return json.dumps({
            "available_slots": [s.to_dict() for s in slots],
            "count": len(slots),
            "instructions": "Present 2-3 slots to the recruiter. When they choose one, call book_meeting with slot_start_iso."
        }, indent=2)

    async def _arun(self, days_ahead: int = 7) -> str:
        return self._run(days_ahead)


# ── Book Meeting Tool ─────────────────────────────────────────────────────────
class BookMeetingTool(BaseTool):
    name: str        = "book_meeting"
    description: str = (
        "Book a confirmed calendar meeting after the recruiter has chosen a time slot. "
        "Requires: slot_start_iso (from check_availability), recruiter_name, recruiter_email. "
        "Cleans up voice-transcribed emails automatically."
    )
    args_schema: Type[BaseModel] = BookMeetingInput

    def _run(
        self,
        slot_start_iso: str,
        recruiter_name: str,
        recruiter_email: str,
        meeting_title: Optional[str] = None,
    ) -> str:
        # Clean email from voice transcription artifacts
        cleaned_email = _clean_email(recruiter_email)

        if not _is_valid_email(cleaned_email):
            return json.dumps({
                "success": False,
                "message": (
                    f"I couldn't parse that email address — I heard '{recruiter_email}'. "
                    f"Could you spell it out slowly? For example: 'john dot smith at gmail dot com'."
                )
            })

        try:
            start = datetime.fromisoformat(slot_start_iso).astimezone(TZ)
        except ValueError as e:
            return json.dumps({"success": False, "message": f"Invalid time format: {e}"})

        slot   = TimeSlot(start=start, end=start + timedelta(minutes=SLOT_DURATION_MIN))
        result = book_meeting(
            slot=slot,
            recruiter_name=recruiter_name.strip(),
            recruiter_email=cleaned_email,
            meeting_title=meeting_title,
        )

        if result["success"]:
            logger.info(f"Meeting booked for {recruiter_name} at {result['slot_display']}")
            return json.dumps({
                "success":      True,
                "message":      (
                    f"Confirmed! Meeting booked for {result['slot_display']} ({start.strftime('%A')}). "
                    f"Calendar invite sent to {cleaned_email}. "
                    f"Always state the exact day name from the slot_display field when confirming."
                ),
                "meet_link":    result.get("meet_link", ""),
                "slot_display": result["slot_display"],
            })
        else:
            logger.error(f"Booking failed: {result.get('error')}")
            return json.dumps({
                "success": False,
                "message": f"Booking failed: {result.get('error', 'Unknown error')}. Please try again.",
            })

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ── Export ────────────────────────────────────────────────────────────────────
CALENDAR_TOOLS = [
    CheckAvailabilityTool(),
    BookMeetingTool(),
]