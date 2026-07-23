"""
The prompt is the strong prior — a tool's return string is only a hint the model
may ignore. That was settled the hard way on July 23: the API and the tool both
told the agent not to propose next year, and it kept doing it, because prompt.py
said otherwise. So the escalation rules have to live here too, and be tested here.

Pure prompt assertions — no livekit import, so these run in an environment where
the rest of the suite cannot.
"""
from prompt import build_greeter_prompt

RESTAURANT = {
    "name": "Test Restaurant",
    "agent_name": "Sophia",
    "timezone": "America/New_York",
    "phone": "+1 407 555 0182",
}
PROMPT = build_greeter_prompt(RESTAURANT)
LOWER = PROMPT.lower()


class TestTransferSectionExists:
    def test_names_both_tools(self):
        assert "transfer_to_human" in PROMPT
        assert "take_message" in PROMPT

    def test_has_its_own_labelled_section(self):
        assert "SPEAKING TO A PERSON" in PROMPT


class TestWhenToEscalate:
    def test_honours_an_explicit_request_without_arguing(self):
        assert "do not argue" in LOWER or "do NOT argue" in PROMPT

    def test_covers_complaints(self):
        assert "unhappy" in LOWER or "complaint" in LOWER

    def test_covers_large_and_private_events(self):
        assert "private event" in LOWER or "large party" in LOWER

    def test_covers_the_agent_being_stuck(self):
        # Basel chose "when asked OR when stuck", so a repeated misunderstanding
        # must be a trigger and not only an explicit request.
        assert "twice" in LOWER


class TestNeverPromiseBeforeTheToolAnswers:
    """The single most damaging failure: the agent says 'putting you through',
    the transfer is not configured, and the caller hears silence."""

    def test_forbids_announcing_a_transfer_first(self):
        assert "never say you are transferring" in LOWER

    def test_says_to_call_the_tool_first(self):
        assert "call the tool first" in LOWER


class TestFallbackIsSpelledOut:
    def test_tells_it_to_collect_name_number_and_reason(self):
        assert "name" in LOWER
        assert "number to reach them" in LOWER

    def test_tells_it_to_read_the_number_back(self):
        assert "read the number back" in LOWER


class TestStaysInCharacter:
    def test_forbids_robotic_words(self):
        assert "supervisor" in LOWER  # named as forbidden
        assert "operator" in LOWER

    def test_offers_natural_phrasing(self):
        assert "put you through" in LOWER

    def test_does_not_break_the_never_mention_ai_rule(self):
        # The persona rule elsewhere in the prompt must still stand.
        assert "Never mention AI, bot, or automated" in PROMPT


class TestDoesNotDisturbExistingBehaviour:
    def test_greeting_and_faq_guidance_survive(self):
        assert "HANDLING QUESTIONS" in PROMPT
        assert "GARBLED OR UNCLEAR SPEECH" in PROMPT

    def test_end_of_call_rules_survive(self):
        assert "ENDING THE CALL (STRICT)" in PROMPT

    def test_agent_name_still_renders(self):
        assert "Sophia" in PROMPT
