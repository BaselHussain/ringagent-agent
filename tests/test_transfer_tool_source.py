"""
Guards the escalation-to-a-human tools in agent.py.

The dangerous failures here are all silent in a unit test but obvious to a real
caller: a tool that isn't registered, a request that 401s because it forgot the
shared secret, a transfer that rings forever because nobody set a timeout, or a
failed transfer that still reports itself as a success (so the call is logged as
"escalated" and the customer never gets a call back).

Parses the source rather than importing agent.py — the module needs
livekit.plugins, which is not installed here (the pre-existing suite has been
failing on that import for a while). Source inspection catches exactly these
mistakes and needs no dependencies.
"""
import ast
import os

AGENT_PY = os.path.join(os.path.dirname(__file__), "..", "agent.py")

with open(AGENT_PY, encoding="utf-8") as f:
    SOURCE = f.read()
TREE = ast.parse(SOURCE)

TRANSFER_TOOLS = ("transfer_to_human", "take_message")


def _func(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _api_calls_in(func_node):
    """httpx .post/.get calls inside one function, keyed off the API base URL."""
    calls = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr in ("get", "post")):
            continue
        if not node.args:
            continue
        url = node.args[0]
        if isinstance(url, ast.JoinedStr) and "RINGAGENT_API_URL" in ast.unparse(url):
            calls.append(node)
    return calls


def test_both_tools_exist_and_are_registered_as_function_tools():
    found = []
    for name in TRANSFER_TOOLS:
        node = _func(name)
        assert node is not None, f"{name} is missing from agent.py"
        decorators = [ast.unparse(d) for d in node.decorator_list]
        assert "function_tool" in decorators, (
            f"{name} is not decorated with @function_tool — the LLM cannot call it. "
            f"Decorators found: {decorators}"
        )
        found.append(name)
    # A matcher that matched nothing would report "all pass" while checking
    # nothing, so print what was actually verified.
    print(f"\nverified {len(found)} escalation tools: {', '.join(found)}")
    assert len(found) == 2


def test_both_tools_send_the_shared_secret():
    checked = 0
    for name in TRANSFER_TOOLS:
        calls = _api_calls_in(_func(name))
        assert calls, f"{name} makes no API call — it cannot do anything"
        for node in calls:
            headers = [kw for kw in node.keywords if kw.arg == "headers"]
            assert headers, f"{name} line {node.lineno}: API call has no headers (will 401 on a live call)"
            assert ast.unparse(headers[0].value) == "API_HEADERS", (
                f"{name} line {node.lineno}: headers must be API_HEADERS, got "
                f"{ast.unparse(headers[0].value)}"
            )
            checked += 1
    print(f"\nchecked {checked} escalation API call sites carry API_HEADERS")
    assert checked >= 2


def test_tools_hit_the_expected_endpoints():
    urls = {
        name: [ast.unparse(c.args[0]) for c in _api_calls_in(_func(name))]
        for name in TRANSFER_TOOLS
    }
    assert any("/agent/request-transfer" in u for u in urls["transfer_to_human"]), urls
    assert any("/agent/escalation" in u for u in urls["take_message"]), urls


def test_transfer_sets_a_ringing_timeout():
    """A blind transfer with no timeout rings forever and the caller never comes
    back to us — the exact failure the message fallback exists to prevent."""
    node = _func("transfer_to_human")
    src = ast.unparse(node)
    assert "ringing_timeout" in src, "transfer_to_human must set ringing_timeout on the REFER"
    assert "transfer_sip_participant" in src


def test_transfer_speaks_before_the_refer_not_after():
    """Once the REFER lands our audio path is gone, so anything said after it is
    silence to the caller."""
    node = _func("transfer_to_human")
    body = ast.unparse(node)
    say_at = body.find("session.say")
    refer_at = body.find("transfer_sip_participant")
    assert say_at != -1, "transfer_to_human never speaks to the caller"
    assert refer_at != -1
    assert say_at < refer_at, "the spoken line must come BEFORE the transfer, not after"


def test_failed_transfer_never_reports_success():
    """_escalated drives the call outcome. Setting it on a failed transfer would
    log the call as handled AND suppress the missed-call text, so the customer
    would be dropped silently."""
    node = _func("transfer_to_human")
    for handler in ast.walk(node):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        assigns = [
            ast.unparse(n)
            for n in ast.walk(handler)
            if isinstance(n, ast.Assign) and "_escalated" in ast.unparse(n.targets[0])
        ]
        assert not assigns, f"_escalated set inside an except block: {assigns}"

    # And it must be set exactly once, on the success path.
    sets_true = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Assign)
        and "_escalated" in ast.unparse(n.targets[0])
        and ast.unparse(n.value) == "True"
    ]
    assert len(sets_true) == 1, f"expected exactly one `_escalated = True`, found {len(sets_true)}"


def test_every_refusal_path_points_at_take_message():
    """If a failure path returns a bare apology the model improvises; every one
    must tell it to take a message instead."""
    node = _func("transfer_to_human")
    returns = [
        ast.unparse(n.value)
        for n in ast.walk(node)
        if isinstance(n, ast.Return) and n.value is not None
    ]
    assert returns, "transfer_to_human returns nothing"
    # The success return is the only one allowed not to mention take_message.
    non_message = [
        r for r in returns if "take_message" not in r and "take_message_instead" not in r
    ]
    print(f"\n{len(returns)} return paths, {len(non_message)} without a take-a-message instruction")
    assert len(non_message) <= 1, (
        "these failure paths leave the agent with no next step:\n" + "\n".join(non_message)
    )


def test_escalated_outcome_is_wired_into_the_call_log():
    """The flag is pointless if _on_call_end never reads it."""
    assert '"escalated" if agent._escalated' in SOURCE, (
        "the end-of-call outcome does not report escalated calls"
    )
    # booked/ordered must still win over escalated.
    booked_at = SOURCE.find('"booked" if agent._reservation_saved')
    escalated_at = SOURCE.find('"escalated" if agent._escalated')
    assert booked_at != -1 and escalated_at != -1
    assert booked_at < escalated_at, "a completed booking must outrank escalated"


def test_sip_participant_is_looked_up_at_transfer_time():
    """Caching the participant at startup is a race — the room is still filling
    when the job begins."""
    node = _func("_find_sip_participant")
    assert node is not None, "_find_sip_participant helper is missing"
    src = ast.unparse(node)
    assert "remote_participants" in src
    assert "sip.phoneNumber" in src
