"""
scripts/setup_vapi.py
─────────────────────────────────────────────────────────────────────────────
One-time Vapi assistant setup script.

Usage:
    python scripts/setup_vapi.py --server-url https://your-domain.com

This will:
  1. Create a Vapi assistant with your server as the LLM backend
  2. Assign a phone number to the assistant
  3. Print the phone number to dial
  4. Save VAPI_ASSISTANT_ID to .env
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key

load_dotenv()

VAPI_API_KEY          = os.getenv("VAPI_API_KEY")
VAPI_PHONE_NUMBER_ID  = os.getenv("VAPI_PHONE_NUMBER_ID")
CANDIDATE_NAME        = os.getenv("CANDIDATE_NAME", "Candidate")
ENV_FILE              = Path(".env")


async def create_assistant(server_url: str) -> dict:
    """Create Vapi assistant pointing at our webhook."""
    payload = {
        "name": f"{CANDIDATE_NAME} AI Persona",
        "model": {
            "provider": "custom-llm",
            "url":       f"{server_url}/voice/webhook",
            "model":     "gpt-4o",    # informational only for custom-llm
        },
        "voice": {
            "provider": "11labs",
            "voiceId":  "rachel",
        },
        "transcriber": {
            "provider": "deepgram",
            "model":    "nova-2",
            "language": "en",
        },
        "firstMessage":           "",
        "endCallMessage":         f"Thank you for speaking with {CANDIDATE_NAME}'s assistant!",
        "serverUrl":              f"{server_url}/voice/webhook",
        "backgroundSound":        "off",
        "silenceTimeoutSeconds":  30,
        "maxDurationSeconds":     600,
        "endCallPhrases":         ["goodbye", "thanks bye", "that's all"],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.vapi.ai/assistant",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


async def assign_phone_number(assistant_id: str) -> dict:
    """Assign the configured phone number to the assistant."""
    if not VAPI_PHONE_NUMBER_ID:
        print("⚠️  VAPI_PHONE_NUMBER_ID not set. Skipping phone assignment.")
        print("   Buy a number in the Vapi dashboard and set VAPI_PHONE_NUMBER_ID.")
        return {}

    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"https://api.vapi.ai/phone-number/{VAPI_PHONE_NUMBER_ID}",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            json={"assistantId": assistant_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


async def get_phone_number_details() -> dict:
    """Fetch phone number details from Vapi."""
    if not VAPI_PHONE_NUMBER_ID:
        return {}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.vapi.ai/phone-number/{VAPI_PHONE_NUMBER_ID}",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


async def main():
    parser = argparse.ArgumentParser(description="Set up Vapi assistant")
    parser.add_argument(
        "--server-url",
        required=True,
        help="Public URL of your FastAPI server (e.g. https://abc.ngrok.io)",
    )
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")

    if not VAPI_API_KEY:
        print("❌ VAPI_API_KEY not set in .env")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Setting up Vapi for: {CANDIDATE_NAME}")
    print(f"  Server URL: {server_url}")
    print(f"{'='*50}\n")

    # 1. Create assistant
    print("1. Creating Vapi assistant...")
    assistant = await create_assistant(server_url)
    assistant_id = assistant["id"]
    print(f"   ✅ Assistant created: {assistant_id}")

    # 2. Save to .env
    if ENV_FILE.exists():
        set_key(str(ENV_FILE), "VAPI_ASSISTANT_ID", assistant_id)
        print(f"   ✅ VAPI_ASSISTANT_ID saved to .env")

    # 3. Assign phone number
    print("\n2. Assigning phone number...")
    await assign_phone_number(assistant_id)

    # 4. Get phone number details
    phone_data = await get_phone_number_details()
    phone_num  = phone_data.get("number", "Check Vapi dashboard")

    print(f"\n{'='*50}")
    print(f"  🎉 Setup complete!")
    print(f"  📞 Phone number: {phone_num}")
    print(f"  🤖 Assistant ID: {assistant_id}")
    print(f"  🌐 Webhook URL:  {server_url}/voice/webhook")
    print(f"{'='*50}\n")
    print("Next steps:")
    print("  1. Call the number above to test")
    print("  2. Update PHONE_NUM in frontend/index.html")
    print("  3. Keep server running!")


if __name__ == "__main__":
    asyncio.run(main())
