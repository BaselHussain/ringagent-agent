"""
Ring Agent — LiveKit voice agent (Sarah the restaurant receptionist)
Architecture: Two agents with handoff per LiveKit SKILL.md guidance.
  GreeterAgent  → handles FAQ/general chat (lean context)
  ReservationAgent → handles booking flow STEP A-D (focused context)
"""
import asyncio
import logging
import os
import time
import httpx
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    UserStateChangedEvent,
    cli,
    inference,
    room_io,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation
from pydantic import BaseModel, Field

from prompt import build_greeter_prompt, build_reservation_prompt, build_order_prompt

load_dotenv()
logger = logging.getLogger("ring-agent")
logging.basicConfig(level=logging.DEBUG)

RINGAGENT_API_URL = os.environ.get("RINGAGENT_API_URL", "")
DEMO_RESTAURANT_ID = os.environ.get("DEMO_RESTAURANT_ID", "86982824-7063-4235-ad95-329e2877f483")
# Platform default voice — used when a restaurant hasn't picked its own voice.
DEFAULT_VOICE_ID = "XrExE9yKIg1WjnnlVkGX"


class OrderItem(BaseModel):
    """One line of a pickup order, exactly as the caller said it. The backend
    matches the name against the real menu and prices it — the agent never
    supplies a price or a total.

    quantity/notes are Optional because GPT sends explicit nulls for fields it
    has no value for — a plain `str`/`int` makes pydantic reject the whole tool
    call (live failure, Hassan's July 17 test call)."""

    name: str = Field(description="The dish as the caller said it, e.g. 'margherita pizza'")
    quantity: int | None = Field(default=1, description="How many of this item (default 1)")
    notes: str | None = Field(default=None, description="Modifications/allergies word-for-word, e.g. 'no onions, peanut allergy'")


def _order_items_payload(items) -> list:
    """OrderItem models (or dicts, defensively) -> plain dicts for the API,
    with nulls normalized so the backend always sees a clean shape."""
    out = []
    for i in items or []:
        if hasattr(i, "model_dump"):
            d = i.model_dump()
        elif isinstance(i, dict):
            d = dict(i)
        else:
            continue
        if d.get("quantity") is None:
            d["quantity"] = 1
        if d.get("notes") is None:
            d["notes"] = ""
        out.append(d)
    return out


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
        self._agent_name = restaurant.get("agent_name") or "Sarah"
        self._reservation_saved = False
        self._order_saved = False
        self._caller_name = ""
        self._lead_reason = ""
        super().__init__(
            instructions=self._build_unified_prompt(restaurant, caller_phone),
            tools=[EndCallTool()],
        )

    def _build_unified_prompt(self, restaurant: dict, caller_phone: str) -> str:
        """Build a unified prompt that handles greeting, FAQ, and reservation."""
        greeter_part = build_greeter_prompt(restaurant)
        reservation_part = build_reservation_prompt(restaurant, caller_phone)
        order_part = build_order_prompt(restaurant)

        return f"""{greeter_part}

---

TRANSITION TO RESERVATION:
When the caller asks to make a reservation (e.g., "I'd like to book", "Can I reserve a table?"):
1. Move immediately to the RESERVATION FLOW below
2. DO NOT greet again or re-introduce yourself
3. SCAN the entire conversation history (from greeting phase) for any information already given
4. Use information from greeting phase in reservation (name, party size, date, time if already stated)

TRANSITION TO PICKUP ORDER:
When the caller wants to order food to pick up (e.g., "I'd like to place an order", "Can I order a pizza for pickup?", "I want to grab some food to go"):
1. Move immediately to the ORDER FLOW below — do not greet again
2. SCAN the conversation history for anything already given (name, items, pickup time)
3. A reservation and an order can BOTH happen on one call — the flows are independent

---

RESERVATION FLOW:
{reservation_part}

---

ORDER FLOW:
{order_part}"""

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
                         f"This is {self._agent_name}, how can I help you today?' — say nothing else."
        )

    @function_tool
    async def check_availability(
        self,
        context: RunContext,
        party_size: str,
        date: str,
        time: str,
    ) -> str:
        """Call this in STEP B.6, right after you have the party size, date AND time — BEFORE special
        requests and BEFORE saving. Confirms the restaurant has room at that time. Say nothing while it runs.
        Follow the instruction it returns. Never save a reservation for a time this tool reports as full.

        Args:
            party_size: Number of people e.g. '4 people'
            date: Full date e.g. 'Friday June 27'
            time: Requested time e.g. '7:00 PM'
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{RINGAGENT_API_URL}/agent/check-availability",
                    json={
                        "restaurant_id": self.restaurant.get("id"),
                        "party_size": party_size,
                        "date": date,
                        "time": time,
                    },
                    timeout=10.0,
                )
            data = resp.json()
        except Exception as e:
            logger.error("check_availability failed (failing open): %s", e)
            return "Availability could not be checked; treat the time as open and continue to special requests."

        # Fail-open: capacity off / not configured.
        if not data.get("enforced", False):
            return "That time is open. Continue to special requests."
        if data.get("available"):
            return "That time is open. Continue to special requests."

        alts = data.get("alternatives") or []
        if alts:
            return (
                f"That time is fully booked. Offer these alternative times to the caller: {', '.join(alts)}. "
                "Ask which they'd like, then call check_availability again with the new time before continuing. "
                "Do not book the original time."
            )
        return (
            "That time is fully booked and nothing nearby is open that day. Apologize and ask the caller "
            "for a different date or time, then call check_availability again with the new details."
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
        """Call ONLY after check_availability (STEP B.6) has confirmed the time is open, the caller has
        explicitly said yes to the full readback of all four details,
        AND after you have already asked for special requests / seating preferences (STEP B.5) and captured them in notes.
        Say nothing while this runs. After it returns, speak the spoken confirmation the tool result tells you to give.

        Args:
            customer_name: The caller's name for the reservation
            party_size: Number of people e.g. '4 people'
            date: Full date e.g. 'Friday June 27'
            time: Reservation time e.g. '7:00 PM'
            notes: Special occasion or requests e.g. 'Birthday, wants a cake'. Empty string if none.
        """
        if self._reservation_saved:
            logger.warning("Duplicate save_reservation call prevented (already saved)")
            return "Reservation already saved."

        logger.info("Saving reservation: %s, %s, %s, %s", customer_name, party_size, date, time)
        self._reservation_saved = True
        self._caller_name = customer_name
        if not self._lead_reason:
            self._lead_reason = "reservation"

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
            "Reservation saved successfully. Now speak a brief, warm confirmation OUT LOUD to the caller: "
            "restate their name, party size, date and time (and the note if there is one), then ask "
            "'Is there anything else I can help you with?' Stay on the line and wait — do NOT end the call."
        )

    @function_tool
    async def lookup_reservation(self, context: RunContext, name: str = "", date: str = "") -> str:
        """Find the caller's existing upcoming reservation(s). Call this FIRST whenever the caller wants
        to change or cancel a reservation. With NO arguments it looks up by the number they're calling
        from. If that finds nothing, ask the caller for the name (and date if they know it) on the booking,
        then call this again with name set to search by name instead.
        Returns the details — including the reservation_id you need — or reports that none were found.

        Args:
            name: The name the booking is under, to search by name (empty to search by caller's phone)
            date: Optional date to narrow a name search e.g. 'Friday July 18' (empty if unknown)
        """
        try:
            async with httpx.AsyncClient() as client:
                if name.strip():
                    r = await client.get(
                        f"{RINGAGENT_API_URL}/agent/reservations-by-name",
                        params={"restaurant_id": self.restaurant.get("id", ""), "name": name, "date": date},
                        timeout=10.0,
                    )
                else:
                    r = await client.get(
                        f"{RINGAGENT_API_URL}/agent/reservations-by-phone/{self.caller_phone}",
                        params={"restaurant_id": self.restaurant.get("id", "")},
                        timeout=10.0,
                    )
                rows = r.json() if r.status_code == 200 else []
        except Exception as e:
            logger.error("lookup_reservation failed: %s", e)
            return "I couldn't look that up just now. Ask the caller for the name and date on the booking."

        if not rows:
            if name.strip():
                return (
                    "No reservation was found under that name. Double-check the name and date with the "
                    "caller; if it's still not found, warmly take their name and number and let them know "
                    "a team member will follow up."
                )
            return (
                "No reservation was found under this phone number. Ask the caller for the name (and date) "
                "the booking is under, then call lookup_reservation again with that name."
            )
        lines = [
            f"reservation_id={row.get('id')}: {row.get('customer_name')}, party of "
            f"{row.get('party_size')}, {row.get('date')} at {row.get('time')} (status: {row.get('status')})"
            for row in rows
        ]
        return (
            "Found these reservation(s). Read the details back so the caller confirms which one, then use "
            "its reservation_id to modify or cancel:\n" + "\n".join(lines)
        )

    @function_tool
    async def modify_reservation(
        self,
        context: RunContext,
        reservation_id: str,
        party_size: str = "",
        date: str = "",
        time: str = "",
    ) -> str:
        """Change an existing reservation. Call ONLY after lookup_reservation and after the caller
        confirms the change. Pass the reservation_id from lookup_reservation and ONLY the fields that change.

        Args:
            reservation_id: The id from lookup_reservation
            party_size: New party size e.g. '4 people' (empty string if unchanged)
            date: New date e.g. 'Saturday July 19' (empty string if unchanged)
            time: New time e.g. '8:00 PM' (empty string if unchanged)
        """
        self._lead_reason = "reservation change"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{RINGAGENT_API_URL}/agent/modify-reservation",
                    json={"reservation_id": reservation_id, "party_size": party_size, "date": date, "time": time},
                    timeout=10.0,
                )
        except Exception as e:
            logger.error("modify_reservation failed: %s", e)
            return "Something went wrong updating that. Tell the caller a team member will follow up."
        return (
            "Reservation updated. Now speak a brief, warm confirmation OUT LOUD restating the new details, "
            "then ask if there's anything else. Stay on the line — do NOT end the call."
        )

    @function_tool
    async def cancel_reservation(self, context: RunContext, reservation_id: str) -> str:
        """Cancel an existing reservation. Call ONLY after lookup_reservation and after the caller
        EXPLICITLY confirms they want to cancel. Pass the reservation_id from lookup_reservation."""
        self._lead_reason = "reservation cancellation"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{RINGAGENT_API_URL}/agent/cancel-reservation",
                    json={"reservation_id": reservation_id},
                    timeout=10.0,
                )
        except Exception as e:
            logger.error("cancel_reservation failed: %s", e)
            return "Something went wrong cancelling that. Tell the caller a team member will follow up."
        return (
            "Reservation cancelled. Now confirm OUT LOUD, warmly, that it's cancelled and ask if there's "
            "anything else. Stay on the line — do NOT end the call."
        )

    @function_tool
    async def quote_order(
        self,
        context: RunContext,
        items: list[OrderItem],
        requested_time: str = "",
    ) -> str:
        """Price a pickup order — STEP O.D, MANDATORY before reading any total back to the caller.
        Pass the CURRENT full list of items every time. Writes nothing; safe to call repeatedly
        after corrections. Say nothing while it runs. Follow the instruction it returns.

        Args:
            items: Every item currently in the order (name as the caller said it, quantity, notes)
            requested_time: The caller's pickup-time words verbatim, e.g. 'in 20 minutes' or '7:30 PM' (empty if they don't mind)
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{RINGAGENT_API_URL}/agent/quote-order",
                    json={
                        "restaurant_id": self.restaurant.get("id"),
                        "items": _order_items_payload(items),
                        "requested_time": requested_time,
                    },
                    timeout=10.0,
                )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
        except Exception as e:
            logger.error("quote_order failed: %s", e)
            return (
                "The order system isn't responding. Apologize and offer to take their name and number "
                "so the restaurant can call them right back. Do not quote any prices."
            )

        if not data.get("enabled", False):
            return (
                "Phone orders aren't available at this restaurant. Apologize warmly and offer to help "
                "with a reservation or a question instead."
            )

        unmatched = data.get("unmatched") or []
        lines = data.get("lines") or []
        if unmatched:
            return (
                f"These items are NOT on the menu: {', '.join(unmatched)}. Tell the caller, suggest the "
                "closest real dishes from the MENU, fix the list with them, then call quote_order again "
                "with the corrected items. Do not read back a total yet."
            )
        if not lines:
            return "No items yet. Ask the caller what they'd like to order from the menu."

        parts = []
        for l in lines:
            piece = f"{l.get('quantity')}x {l.get('name')}"
            if l.get("line_total") is None:
                piece += " (priced at pickup)"
            else:
                piece += f" (${l.get('line_total'):.2f})"
            if l.get("notes"):
                piece += f" [note: {l.get('notes')}]"
            parts.append(piece)
        total_bit = f"Total ${data.get('total', 0):.2f}"
        if data.get("tax"):
            total_bit += f" (includes ${data.get('tax'):.2f} tax)"
        if data.get("has_unpriced"):
            total_bit += " plus market-price items confirmed at pickup"
        time_bit = f"Ready around {data.get('ready_text')}."
        if data.get("adjusted"):
            time_bit = (
                f"The caller's requested time is sooner than the kitchen can manage — offer {data.get('ready_text')} "
                "instead ('The kitchen can have that ready around that time — does that work?')."
            )
        return (
            f"Order priced: {'; '.join(parts)}. {total_bit}. {time_bit} "
            "Now read the full order, total and pickup time back to the caller and get an explicit yes "
            "before calling save_order. Say the prices naturally (e.g. 'thirty-four fifty-six')."
        )

    @function_tool
    async def save_order(
        self,
        context: RunContext,
        customer_name: str,
        items: list[OrderItem],
        requested_time: str = "",
        notes: str = "",
    ) -> str:
        """Save the pickup order — STEP O.F, ONLY after quote_order succeeded and the caller gave an
        explicit yes to the full readback. Pass the SAME confirmed items. Say nothing while it runs.
        Follow the instruction it returns exactly.

        Args:
            customer_name: Name for the order
            items: The confirmed items (same list the caller said yes to)
            requested_time: The caller's pickup-time words verbatim (same as quoted)
            notes: Any order-wide note that isn't tied to one item (empty if none)
        """
        if self._order_saved:
            return (
                "The order is already saved. If the caller wants to change it now, apologize and say "
                "they can mention it at pickup or a team member will adjust it — do not save again."
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{RINGAGENT_API_URL}/agent/save-order",
                    json={
                        "restaurant_id": self.restaurant.get("id"),
                        "caller_phone": self.caller_phone,
                        "customer_name": customer_name,
                        "items": _order_items_payload(items),
                        "requested_time": requested_time,
                        "notes": notes,
                    },
                    timeout=10.0,
                )
        except Exception as e:
            logger.error("save_order failed: %s", e)
            return (
                "The order could NOT be saved. Do NOT tell the caller it was placed. Apologize and take "
                "their name and number so the restaurant can call them back to confirm the order."
            )

        if resp.status_code != 200:
            logger.error("save_order rejected: HTTP %s %s", resp.status_code, resp.text[:200])
            try:
                err = (resp.json() or {}).get("error", "")
            except Exception:
                err = ""
            if err == "no_matched_items":
                return (
                    "None of those items matched the menu, so nothing was saved. Go back to the order, "
                    "fix the items with the caller, and call quote_order again."
                )
            return (
                "The order could NOT be saved. Do NOT tell the caller it was placed. Apologize and take "
                "their name and number so the restaurant can call them back to confirm the order."
            )

        # Confirmed success only past this point (unlike save_reservation's flag-first pattern).
        self._order_saved = True
        if customer_name:
            self._caller_name = customer_name
        if not self._lead_reason or self._lead_reason == "inbound call":
            self._lead_reason = "pickup order"

        data = resp.json()
        total_text = data.get("total_text") or f"${data.get('total', 0):.2f}"
        ready_text = data.get("ready_text") or "shortly"
        return (
            f"Order saved. Total {total_text}, ready around {ready_text}. Now speak a brief, warm "
            "confirmation OUT LOUD: the total, that it'll be ready around that time, and that they pay "
            "at pickup. Then ask 'Is there anything else I can help you with?' and STAY on the line — "
            "do NOT end the call."
        )

    @function_tool
    async def capture_lead(self, context: RunContext, name: str = "", reason: str = "") -> str:
        """Silently record who is calling and why, for the restaurant's records. Call this once, early,
        as soon as you naturally know the caller's name and/or why they're calling (a booking, a question,
        a change, an event, etc.). Say NOTHING out loud when calling this — it's a background note.

        Args:
            name: The caller's name if known (empty string if not)
            reason: A short reason for the call e.g. 'reservation', 'asking about hours', 'private event'
        """
        if name:
            self._caller_name = name
        if reason:
            self._lead_reason = reason
        return "Noted silently. Say nothing about this to the caller; continue naturally."


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
        tts=inference.TTS(
            "elevenlabs/eleven_turbo_v2_5",
            voice=(restaurant.get("voice_id") or DEFAULT_VOICE_ID),
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "min_delay": 0.2,
                "max_delay": 0.8,
            },
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
        # Mark the caller "away" after this many seconds of total silence.
        user_away_timeout=10.0,
    )

    agent = SarahAgent(restaurant, caller_phone)
    call_start = time.monotonic()

    # When the call ends: record the call (so it's counted + has a transcript)
    # AND log a lead (converted=True if a booking or order was made this call).
    async def _on_call_end() -> None:
        # "booked" and "ordered" are both SUCCESS_OUTCOMES on the API side; a
        # call with both keeps "booked" (one value per call, either suppresses
        # the missed-call text).
        outcome = (
            "booked" if agent._reservation_saved
            else "ordered" if agent._order_saved
            else "other"
        )
        duration = int(time.monotonic() - call_start)
        # Best-effort transcript from the conversation history.
        transcript = ""
        try:
            parts = []
            for item in session.history.items:
                role = getattr(item, "role", "") or ""
                text = getattr(item, "text_content", None)
                if text is None:
                    text = getattr(item, "content", "")
                if isinstance(text, (list, tuple)):
                    text = " ".join(str(t) for t in text)
                text = (str(text) if text else "").strip()
                if role in ("user", "assistant") and text:
                    speaker = "Caller" if role == "user" else agent._agent_name
                    parts.append(f"{speaker}: {text}")
            transcript = "\n".join(parts)
        except Exception as e:
            logger.error("Transcript build failed: %s", e)

        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{RINGAGENT_API_URL}/agent/end-of-call",
                    json={
                        "restaurant_id": restaurant.get("id"),
                        "caller_phone": caller_phone,
                        "duration": duration,
                        "transcript": transcript,
                        "outcome": outcome,
                    },
                    timeout=10.0,
                )
            except Exception as e:
                logger.error("Call log failed: %s", e)
            try:
                await client.post(
                    f"{RINGAGENT_API_URL}/agent/lead",
                    json={
                        "restaurant_id": restaurant.get("id"),
                        "name": agent._caller_name,
                        "phone": caller_phone,
                        "reason": agent._lead_reason or "inbound call",
                        "converted": agent._reservation_saved or agent._order_saved,
                    },
                    timeout=10.0,
                )
            except Exception as e:
                logger.error("Lead capture failed: %s", e)

    ctx.add_shutdown_callback(_on_call_end)

    # Hang up on a silent/abandoned call: one gentle check-in, then a friendly
    # sign-off and disconnect — so the line doesn't stay open on dead air.
    inactivity_task: "asyncio.Task | None" = None

    async def _check_if_present() -> None:
        try:
            await session.generate_reply(
                instructions="The caller has gone quiet. Warmly check if they're still "
                             "there in ONE short line (e.g. 'Are you still there?'). Say nothing else."
            )
            await asyncio.sleep(12)
            await session.generate_reply(
                instructions="The caller is still silent. Say ONE short, friendly goodbye — "
                             "'I'll let you go — feel free to call back anytime. Take care!' — and nothing else."
            )
            await asyncio.sleep(4)  # let the farewell finish playing before we cut the line
        except asyncio.CancelledError:
            return
        await session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        nonlocal inactivity_task
        if ev.new_state == "away":
            if inactivity_task is None or inactivity_task.done():
                inactivity_task = asyncio.create_task(_check_if_present())
            return
        # Caller spoke again — cancel the pending check-in/hang-up.
        if inactivity_task is not None:
            inactivity_task.cancel()
            inactivity_task = None

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
