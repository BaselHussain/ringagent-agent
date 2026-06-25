"""
Ring Agent — LiveKit voice agent (Sarah the restaurant receptionist)
Architecture: Two agents with handoff per LiveKit SKILL.md guidance.
  GreeterAgent  → handles FAQ/general chat (lean context)
  ReservationAgent → handles booking flow STEP A-D (focused context)
"""
import logging
import os
import httpx
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation

from prompt import build_greeter_prompt, build_reservation_prompt

load_dotenv()
logger = logging.getLogger("ring-agent")

RINGAGENT_API_URL = os.environ.get("RINGAGENT_API_URL", "")
DEMO_RESTAURANT_ID = os.environ.get("DEMO_RESTAURANT_ID", "86982824-7063-4235-ad95-329e2877f483")


# ---------------------------------------------------------------------------
# Unified Sarah Agent — Greeting + Reservation handling in one agent
# This ensures full conversation history is always available
# ---------------------------------------------------------------------------

class SarahAgent(Agent):
    def __init__(self, restaurant: dict, caller_phone: str) -> None:
        self.restaurant = restaurant
        self.caller_phone = caller_phone
        restaurant_name = restaurant.get("name", "the restaurant")
        self._restaurant_name = restaurant_name
        super().__init__(
            instructions=self._build_unified_prompt(restaurant, caller_phone),
            tools=[EndCallTool()],
        )

    def _build_unified_prompt(self, restaurant: dict, caller_phone: str) -> str:
        """Build a unified prompt that handles greeting, FAQ, and reservation."""
        greeter_part = build_greeter_prompt(restaurant)
        reservation_part = build_reservation_prompt(restaurant, caller_phone)

        return f"""{greeter_part}

---

TRANSITION TO RESERVATION:
When the caller asks to make a reservation (e.g., "I'd like to book", "Can I reserve a table?"):
1. Move immediately to the RESERVATION FLOW below
2. DO NOT greet again or re-introduce yourself
3. SCAN the entire conversation history (from greeting phase) for any information already given
4. Use information from greeting phase in reservation (name, party size, date, time if already stated)

---

RESERVATION FLOW:
{reservation_part}"""

    async def on_enter(self) -> None:
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        timezone = self.restaurant.get("timezone", "America/New_York")
        try:
            now = datetime.now(ZoneInfo(timezone))
            hour = now.hour
        except Exception:
            hour = datetime.now().hour
        time_of_day = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"

        self.session.generate_reply(
            instructions=f"Say exactly this greeting: "
                         f"'Good {time_of_day}, thank you for calling {self._restaurant_name}. "
                         f"This is Sarah, how can I help you today?' — say nothing else."
        )

    @function_tool
    async def save_reservation(
        self,
        context: RunContext,
        customer_name: str,
        party_size: str,
        date: str,
        time: str,
        notes: str = "",
    ) -> str:
        """Call ONLY after the caller has explicitly said yes to the full readback of all four details.
        Say ZERO words before calling this. After it returns, say ZERO words — confirmation is already delivered.

        Args:
            customer_name: The caller's name for the reservation
            party_size: Number of people e.g. '4 people'
            date: Full date e.g. 'Friday June 27'
            time: Reservation time e.g. '7:00 PM'
            notes: Special occasion or requests e.g. 'Birthday, wants a cake'. Empty string if none.
        """
        logger.info("Saving reservation: %s, %s, %s, %s", customer_name, party_size, date, time)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{RINGAGENT_API_URL}/agent/save-reservation",
                    json={
                        "restaurant_id": self.restaurant.get("id"),
                        "caller_phone": self.caller_phone,
                        "customer_name": customer_name,
                        "party_size": party_size,
                        "date": date,
                        "time": time,
                        "notes": notes,
                    },
                    timeout=10.0,
                )
        except Exception as e:
            logger.error("Failed to save reservation: %s", e)
        return (
            "Reservation saved. The caller has already heard the confirmation message "
            "and been asked if they need anything else. Say nothing."
        )


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------

async def _fetch_restaurant(called_number: str | None) -> dict:
    """Fetch restaurant by called phone number, fall back to demo."""
    async with httpx.AsyncClient() as client:
        if called_number:
            try:
                r = await client.get(
                    f"{RINGAGENT_API_URL}/agent/restaurant-by-phone/{called_number}",
                    timeout=5.0,
                )
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        # Fallback to demo restaurant
        r = await client.get(
            f"{RINGAGENT_API_URL}/agent/restaurant/{DEMO_RESTAURANT_ID}",
            timeout=5.0,
        )
        return r.json()


server = AgentServer()


@server.rtc_session(agent_name="ring-agent")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    await ctx.connect()

    # Read SIP participant attributes to get caller + called numbers
    # Attribute names per docs.livekit.io/reference/telephony/sip-participant/
    caller_phone = "unknown"
    called_number = None
    for participant in ctx.room.remote_participants.values():
        attrs = participant.attributes or {}
        if attrs.get("sip.phoneNumber"):
            caller_phone = attrs["sip.phoneNumber"]
        if attrs.get("sip.trunkPhoneNumber"):
            called_number = attrs["sip.trunkPhoneNumber"]
    logger.info("Call from %s to %s", caller_phone, called_number)

    restaurant = await _fetch_restaurant(called_number)

    session = AgentSession(
        stt=inference.STT("deepgram/nova-2-phonecall", language="en"),
        llm=inference.LLM("openai/gpt-4o", extra_kwargs={"temperature": 0.5}),
        tts=inference.TTS("elevenlabs/eleven_turbo_v2_5", voice="XrExE9yKIg1WjnnlVkGX"),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            interruption={
                "mode": "adaptive",
                "min_duration": 0.5,
                "false_interruption_timeout": 2.0,
                "resume_false_interruption": True,
            },
            preemptive_generation={
                "enabled": True,
            },
        ),
    )

    await session.start(
        agent=SarahAgent(restaurant, caller_phone),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
