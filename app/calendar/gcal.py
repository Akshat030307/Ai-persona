"""
calendar/gcal.py
─────────────────────────────────────────────────────────────────────────────
Google Calendar integration.

Provides:
  - get_available_slots(days_ahead)  → list of free time windows
  - book_meeting(slot, recruiter)    → creates event, returns confirmation

OAuth2 flow:
  Run `python -m app.calendar.gcal --auth` once locally to generate token.json.
  Then deploy token.json alongside the app (or use a service account for prod).
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SCOPES               = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_PATH     = os.getenv("GOOGLE_CREDENTIALS_PATH", "./data/google_credentials.json")
TOKEN_PATH           = os.getenv("GOOGLE_TOKEN_PATH", "./data/google_token.json")
CALENDAR_ID          = os.getenv("GOOGLE_CALENDAR_ID", "primary")
CANDIDATE_NAME       = os.getenv("CANDIDATE_NAME", "Candidate")
CANDIDATE_EMAIL      = os.getenv("CANDIDATE_EMAIL", "")
SLOT_DURATION_MIN    = int(os.getenv("BOOKING_SLOT_DURATION_MINUTES", "30"))
DAYS_AHEAD           = int(os.getenv("BOOKING_DAYS_AHEAD", "7"))
DAILY_START_HOUR     = int(os.getenv("BOOKING_DAILY_START_HOUR", "10"))
DAILY_END_HOUR       = int(os.getenv("BOOKING_DAILY_END_HOUR", "18"))
TZ_STR               = os.getenv("BOOKING_TIMEZONE", "Asia/Kolkata")
TZ                   = ZoneInfo(TZ_STR)


# ── Auth ──────────────────────────────────────────────────────────────────────
def get_credentials() -> Credentials:
    """Load or refresh OAuth2 credentials."""
    creds = None
    token_path = Path(TOKEN_PATH)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This branch only runs during initial local setup
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())

    return creds


def get_service():
    """Return an authenticated Google Calendar API service client."""
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds)


# ── Slot Model ────────────────────────────────────────────────────────────────
class TimeSlot:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end   = end

    def to_dict(self) -> dict:
        return {
            "start":         self.start.isoformat(),
            "end":           self.end.isoformat(),
            "display":       self._display(),
            "start_iso":     self.start.isoformat(),
        }

    def _display(self) -> str:
        day  = self.start.strftime("%A, %B %d")
        time = self.start.strftime("%I:%M %p")
        tz   = self.start.strftime("%Z")
        return f"{day} at {time} {tz}"

    def __repr__(self):
        return f"TimeSlot({self._display()})"


# ── Get Available Slots ───────────────────────────────────────────────────────
def get_available_slots(days_ahead: int = DAYS_AHEAD, max_slots: int = 6) -> List[TimeSlot]:
    """
    Check the candidate's calendar and return free SLOT_DURATION_MIN windows.
    Slots are only offered within DAILY_START_HOUR–DAILY_END_HOUR on weekdays.
    """
    service = get_service()
    now     = datetime.now(tz=TZ)
    # Start from next hour to avoid proposing slots in the past
    search_start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    search_end   = search_start + timedelta(days=days_ahead)

    # Fetch existing events in the window (busy times)
    body = {
        "timeMin": search_start.isoformat(),
        "timeMax": search_end.isoformat(),
        "timeZone": TZ_STR,
        "items": [{"id": CALENDAR_ID}],
    }

    try:
        freebusy = service.freebusy().query(body=body).execute()
        busy_periods = freebusy["calendars"][CALENDAR_ID]["busy"]
    except HttpError as e:
        logger.error(f"Google Calendar freebusy error: {e}")
        return []

    # Parse busy periods into datetime ranges
    busy_ranges = []
    for period in busy_periods:
        busy_start = datetime.fromisoformat(period["start"]).astimezone(TZ)
        busy_end   = datetime.fromisoformat(period["end"]).astimezone(TZ)
        busy_ranges.append((busy_start, busy_end))

    # Generate candidate slots (every SLOT_DURATION_MIN within working hours)
    available = []
    cursor    = search_start

    while cursor < search_end and len(available) < max_slots:
        # Skip weekends
        if cursor.weekday() >= 5:
            cursor += timedelta(days=1)
            cursor  = cursor.replace(hour=DAILY_START_HOUR, minute=0, second=0)
            continue

        # Skip outside working hours
        if cursor.hour < DAILY_START_HOUR:
            cursor = cursor.replace(hour=DAILY_START_HOUR, minute=0)
            continue
        if cursor.hour >= DAILY_END_HOUR:
            cursor += timedelta(days=1)
            cursor  = cursor.replace(hour=DAILY_START_HOUR, minute=0, second=0)
            continue

        slot_end = cursor + timedelta(minutes=SLOT_DURATION_MIN)

        # Check if this slot overlaps with any busy period
        is_free = all(
            slot_end <= busy_start or cursor >= busy_end
            for busy_start, busy_end in busy_ranges
        )

        if is_free:
            available.append(TimeSlot(start=cursor, end=slot_end))

        cursor += timedelta(minutes=SLOT_DURATION_MIN)

    logger.info(f"Found {len(available)} available slots")
    return available


# ── Book Meeting ──────────────────────────────────────────────────────────────
def book_meeting(
    slot: TimeSlot,
    recruiter_name: str,
    recruiter_email: str,
    meeting_title: Optional[str] = None,
) -> dict:
    """
    Create a Google Calendar event and send invite to the recruiter.
    Returns confirmation dict with event link and details.
    """
    service = get_service()
    title   = meeting_title or f"Interview: {CANDIDATE_NAME} ↔ {recruiter_name}"

    event = {
        "summary":     title,
        "description": (
            f"Interview meeting between {CANDIDATE_NAME} and {recruiter_name} "
            f"for the AI Engineer role at Scaler.\n\n"
            f"Booked via AI persona assistant."
        ),
        "start": {
            "dateTime": slot.start.isoformat(),
            "timeZone": TZ_STR,
        },
        "end": {
            "dateTime": slot.end.isoformat(),
            "timeZone": TZ_STR,
        },
        "attendees": [
            {"email": CANDIDATE_EMAIL, "displayName": CANDIDATE_NAME},
            {"email": recruiter_email, "displayName": recruiter_name},
        ],
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email",  "minutes": 60},
                {"method": "popup",  "minutes": 15},
            ],
        },
        "conferenceData": {
            "createRequest": {
                "requestId": f"meeting-{slot.start.strftime('%Y%m%d%H%M')}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }

    try:
        created = service.events().insert(
            calendarId=CALENDAR_ID,
            body=event,
            sendUpdates="all",          # sends email invites to all attendees
            conferenceDataVersion=1,    # creates Google Meet link
        ).execute()

        meet_link = (
            created.get("conferenceData", {})
            .get("entryPoints", [{}])[0]
            .get("uri", "")
        )

        logger.info(f"Meeting booked: {created['id']} — {slot.display}")

        return {
            "success":        True,
            "event_id":       created["id"],
            "event_link":     created.get("htmlLink", ""),
            "meet_link":      meet_link,
            "slot_display":   slot._display(),
            "start_iso":      slot.start.isoformat(),
            "recruiter_name": recruiter_name,
            "recruiter_email":recruiter_email,
            "title":          title,
        }

    except HttpError as e:
        logger.error(f"Failed to book meeting: {e}")
        return {
            "success": False,
            "error":   str(e),
        }


# ── Quick Auth Setup (run once locally) ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--auth" in sys.argv:
        print("Running OAuth2 flow to generate token.json...")
        get_credentials()
        print(f"Token saved to {TOKEN_PATH}")
    else:
        # Quick test
        print("Fetching available slots...")
        slots = get_available_slots(days_ahead=3)
        for s in slots:
            print(f"  {s}")
