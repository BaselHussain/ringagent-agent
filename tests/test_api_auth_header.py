"""
Guards the shared secret the backend now requires on /agent/*.

A missed call site would not fail any existing test -- it would fail silently in
production as a 401 during a live phone call, which is the worst possible place
to find out. So this walks agent.py's syntax tree and proves that EVERY request
to the API carries the auth header.

Deliberately parses the source instead of importing agent.py: the module needs
livekit.plugins, which is not installed in this environment (the rest of the
suite has been failing on that import for a while). Source inspection catches
exactly the mistake we care about and needs no dependencies at all.
"""
import ast
import os

AGENT_PY = os.path.join(os.path.dirname(__file__), "..", "agent.py")

with open(AGENT_PY, encoding="utf-8") as f:
    SOURCE = f.read()
TREE = ast.parse(SOURCE)


def _api_calls():
    """Every httpx client.get/.post whose URL is built from RINGAGENT_API_URL."""
    found = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ("get", "post")):
            continue
        # os.environ.get("RINGAGENT_API_URL", "") is also a .get whose argument
        # mentions the constant -- it is config, not a request.
        if ast.unparse(func).startswith("os.environ"):
            continue
        if not node.args:
            continue
        url = node.args[0]
        # Every real call site builds its URL as an f-string on the API base.
        if isinstance(url, ast.JoinedStr) and "RINGAGENT_API_URL" in ast.unparse(url):
            found.append(node)
    return found


def test_finds_the_api_call_sites():
    # A matcher that matches nothing would report "all pass" while checking
    # nothing, so assert the count is plausible and print it.
    calls = _api_calls()
    print(f"\nchecked {len(calls)} API call sites in agent.py")
    assert len(calls) >= 12, f"expected at least 12 API calls, found {len(calls)}"


def test_every_api_call_sends_the_auth_header():
    missing = []
    for node in _api_calls():
        kwargs = {kw.arg for kw in node.keywords}
        if "headers" not in kwargs:
            missing.append(f"line {node.lineno}: {ast.unparse(node.args[0])}")
    assert not missing, "API calls missing the auth header:\n" + "\n".join(missing)


def test_header_value_comes_from_the_shared_constant():
    # Each call must use API_HEADERS, not a hand-rolled dict that could drift.
    wrong = []
    for node in _api_calls():
        for kw in node.keywords:
            if kw.arg == "headers" and ast.unparse(kw.value) != "API_HEADERS":
                wrong.append(f"line {node.lineno}: {ast.unparse(kw.value)}")
    assert not wrong, "API calls not using API_HEADERS:\n" + "\n".join(wrong)


def test_secret_is_read_from_the_environment_and_never_hardcoded():
    assert 'os.environ.get("RINGAGENT_API_SECRET", "")' in SOURCE


def test_no_header_is_sent_when_the_secret_is_unset():
    """An empty secret must produce an empty dict, not a header with a blank
    value -- the backend treats a blank header as a failed match, so local dev
    against an unsecured API would break."""
    namespace = {}
    for node in TREE.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "API_HEADERS":
            # Re-evaluate the real expression with the secret unset.
            expr = ast.unparse(node.value)
            namespace["RINGAGENT_API_SECRET"] = ""
            assert eval(expr, {}, namespace) == {}
            namespace["RINGAGENT_API_SECRET"] = "abc123"
            assert eval(expr, {}, namespace) == {"X-Ringagent-Secret": "abc123"}
            return
    raise AssertionError("API_HEADERS assignment not found in agent.py")
