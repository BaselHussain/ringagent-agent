"""
Mandatory tests per LiveKit SKILL.md — every agent implementation must have tests.
Tests cover: greeting, tool invocation, handoff trigger, reservation flow guard.

NOTE: These are unit tests for agent logic. Integration tests (real SIP calls)
must be done manually using the end-to-end test checklist in the plan.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prompt import build_greeter_prompt, build_reservation_prompt

DEMO_RESTAURANT = {
    "id": "86982824-7063-4235-ad95-329e2877f483",
    "name": "Test Restaurant",
    "address": "123 Main St",
    "phone": "555-1234",
    "hours": {"Monday": "11am-10pm", "Tuesday": "11am-10pm"},
    "timezone": "America/New_York",
    "faqs": [
        {"question": "Do you have parking?", "answer": "Yes, free parking on site."}
    ],
}
DEMO_CALLER = "+12125551234"


# ---------------------------------------------------------------------------
# Prompt builder tests — no LiveKit dependency
# ---------------------------------------------------------------------------

class TestGreeterPrompt:
    def test_contains_restaurant_name(self):
        prompt = build_greeter_prompt(DEMO_RESTAURANT)
        assert "Test Restaurant" in prompt

    def test_contains_hours(self):
        prompt = build_greeter_prompt(DEMO_RESTAURANT)
        assert "Monday" in prompt
        assert "11am-10pm" in prompt

    def test_contains_faq(self):
        prompt = build_greeter_prompt(DEMO_RESTAURANT)
        assert "Do you have parking?" in prompt
        assert "free parking" in prompt

    def test_banned_phrases_listed(self):
        prompt = build_greeter_prompt(DEMO_RESTAURANT)
        assert "one moment" in prompt
        assert "BANNED" in prompt

    def test_handoff_instruction_present(self):
        prompt = build_greeter_prompt(DEMO_RESTAURANT)
        assert "start_reservation" in prompt

    def test_no_booking_steps_in_greeter(self):
        prompt = build_greeter_prompt(DEMO_RESTAURANT)
        # Greeter should NOT contain the reservation steps — that's ReservationAgent's job
        assert "STEP A" not in prompt
        assert "STEP C" not in prompt


class TestReservationPrompt:
    def test_contains_restaurant_id(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert DEMO_RESTAURANT["id"] in prompt

    def test_contains_caller_phone(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert DEMO_CALLER in prompt

    def test_contains_all_steps(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert "STEP A" in prompt
        assert "STEP B" in prompt
        assert "STEP C" in prompt
        assert "STEP D" in prompt

    def test_no_function_call_before_yes(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert "explicit yes" in prompt
        assert "Do NOT call" in prompt or "do not call" in prompt.lower()

    def test_silence_after_save(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert "ZERO words" in prompt

    def test_name_assumption_guard(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert "ONLY if the caller explicitly" in prompt or "explicitly" in prompt

    def test_special_occasion_step(self):
        prompt = build_reservation_prompt(DEMO_RESTAURANT, DEMO_CALLER)
        assert "STEP B.5" in prompt
        assert "MANDATORY" in prompt


# ---------------------------------------------------------------------------
# Agent class tests — mock LiveKit session to test logic
# ---------------------------------------------------------------------------

class TestGreeterAgent:
    def _make_agent(self):
        from agent import GreeterAgent
        return GreeterAgent(DEMO_RESTAURANT, DEMO_CALLER)

    def test_instantiates_correctly(self):
        agent = self._make_agent()
        assert agent.restaurant == DEMO_RESTAURANT
        assert agent.caller_phone == DEMO_CALLER

    @pytest.mark.asyncio
    async def test_start_reservation_returns_reservation_agent(self):
        from agent import GreeterAgent, ReservationAgent
        agent = GreeterAgent(DEMO_RESTAURANT, DEMO_CALLER)
        mock_context = MagicMock()
        result = await agent.start_reservation(mock_context)
        assert isinstance(result, ReservationAgent)
        assert result.restaurant == DEMO_RESTAURANT
        assert result.caller_phone == DEMO_CALLER


class TestReservationAgent:
    def _make_agent(self):
        from agent import ReservationAgent
        return ReservationAgent(DEMO_RESTAURANT, DEMO_CALLER)

    def test_instantiates_correctly(self):
        agent = self._make_agent()
        assert agent.restaurant == DEMO_RESTAURANT
        assert agent.caller_phone == DEMO_CALLER

    @pytest.mark.asyncio
    async def test_save_reservation_calls_api(self):
        from agent import ReservationAgent
        agent = ReservationAgent(DEMO_RESTAURANT, DEMO_CALLER)
        mock_context = MagicMock()

        with patch("agent.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(status_code=200)

            result = await agent.save_reservation(
                mock_context,
                customer_name="Alex",
                party_size="4 people",
                date="Friday June 27",
                time="7:00 PM",
                notes="Birthday",
            )

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert body["customer_name"] == "Alex"
        assert body["restaurant_id"] == DEMO_RESTAURANT["id"]
        assert body["caller_phone"] == DEMO_CALLER

    @pytest.mark.asyncio
    async def test_save_reservation_returns_silence_instruction(self):
        from agent import ReservationAgent
        agent = ReservationAgent(DEMO_RESTAURANT, DEMO_CALLER)
        mock_context = MagicMock()

        with patch("agent.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(status_code=200)

            result = await agent.save_reservation(
                mock_context,
                customer_name="Alex",
                party_size="2",
                date="Saturday",
                time="8pm",
            )

        # Result must instruct the LLM to stay silent
        assert "Say nothing" in result or "say nothing" in result.lower()
