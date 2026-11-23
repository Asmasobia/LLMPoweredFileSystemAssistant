"""
Tests for the agent loop in `llm_file_assistant.run_assistant`, with no network.

A fake client stands in for Groq. It replays a script of model responses and
records every request it receives, so a test can say "the model asks for
list_files, then read_file, then answers" and check what the loop did with each
step: which tools ran, what went back to the model, and whether the history it
built is one the real API would accept.

That last check, `assert_history_is_valid`, carries most of the weight. The API
rejects a conversation in which any assistant tool call lacks a matching tool
reply, and it rejects it on *every later turn*, so one unanswered call breaks the
whole session. A loop can look fine on the turn where it goes wrong and fail only
on the next one. So the tests check the history the API would see, not just the
returned string.
"""

import copy
import json
from types import SimpleNamespace

import pytest

import llm_file_assistant as agent


# ─────────────────────────────────────────────────────────────────────────────
#  Fake LLM client
# ─────────────────────────────────────────────────────────────────────────────
def tool_call(name, arguments, call_id):
    """One tool call, shaped like a Groq SDK object. `arguments` is the raw string."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def wants_tools(*calls):
    return SimpleNamespace(content=None, tool_calls=list(calls))


def answers(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeClient:
    """Replays scripted messages; records a deep copy of every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Deep copy, because the loop keeps appending to the same list. Without
        # the copy every recorded request would show the final history.
        self.requests.append(copy.deepcopy(kwargs))
        if not self.script:
            raise AssertionError("The loop made more model calls than the test scripted.")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(choices=[SimpleNamespace(message=step)])


class LoopingClient(FakeClient):
    """A model that never stops asking for tools."""

    def __init__(self):
        super().__init__([])

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        n = len(self.requests)
        msg = wants_tools(tool_call("list_files", {"directory": "docs"}, f"call_{n}"))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def assert_history_is_valid(messages):
    """Every assistant tool call is answered by exactly one tool message, in order."""
    i = 0
    while i < len(messages):
        m = messages[i]
        i += 1
        if m["role"] == "tool":
            raise AssertionError(f"Tool reply with no preceding tool call: {m}")
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        expected = [tc["id"] for tc in m["tool_calls"]]
        replies = []
        while i < len(messages) and messages[i]["role"] == "tool":
            replies.append(messages[i]["tool_call_id"])
            i += 1
        assert replies == expected, f"Tool calls {expected} answered by {replies}"


def tool_replies(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_ROOT", str(tmp_path))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.txt").write_text("Python developer\nKafka and Redis\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def history():
    return [{"role": "system", "content": agent.SYSTEM_PROMPT}]


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-step chains
# ─────────────────────────────────────────────────────────────────────────────
def test_plain_answer_needs_one_model_call(workspace, history):
    client = FakeClient([answers("Hello.")])

    assert agent.run_assistant("hi", history, client=client) == "Hello."
    assert len(client.requests) == 1
    assert history[-1] == {"role": "assistant", "content": "Hello."}


def test_agent_chains_three_tool_steps_before_answering(workspace, history):
    """
    list -> read -> write -> answer. The old loop forced a no-tools reply after
    the first round of tool calls and returned, so it could never get past
    list_files. This is the test that fails against that version.
    """
    client = FakeClient([
        wants_tools(tool_call("list_files", {"directory": "docs"}, "c1")),
        wants_tools(tool_call("read_file", {"filepath": "docs/alpha.txt"}, "c2")),
        wants_tools(tool_call(
            "write_file", {"filepath": "output/summary.md", "content": "Python, Kafka"}, "c3"
        )),
        answers("Wrote output/summary.md."),
    ])

    reply = agent.run_assistant("Summarise the doc into output/summary.md", history, client=client)

    assert reply == "Wrote output/summary.md."
    assert (workspace / "output" / "summary.md").read_text(encoding="utf-8") == "Python, Kafka"
    assert len(client.requests) == 4
    # Every request offered tools. None was a forced text-only follow-up.
    assert all(r.get("tools") == agent.TOOLS for r in client.requests)
    # Each step saw the previous step's result.
    second_request_tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert "alpha.txt" in second_request_tool_msgs[-1]["content"]
    assert_history_is_valid(history)


def test_several_tool_calls_in_one_response_are_all_answered(workspace, history):
    client = FakeClient([
        wants_tools(
            tool_call("search_in_file", {"filepath": "docs/alpha.txt", "keyword": "kafka"}, "a"),
            tool_call("search_in_file", {"filepath": "docs/alpha.txt", "keyword": "java"}, "b"),
        ),
        answers("Kafka yes, Java no."),
    ])

    agent.run_assistant("Kafka or Java?", history, client=client)

    assert [r["match_count"] for r in tool_replies(history)] == [1, 0]
    assert_history_is_valid(history)


# ─────────────────────────────────────────────────────────────────────────────
#  Bad tool calls come back as errors and the session continues
# ─────────────────────────────────────────────────────────────────────────────
BAD_CALLS = [
    pytest.param("delete_everything", '{"path": "/"}', "Unknown tool", id="unknown-tool"),
    pytest.param("read_file", '{"filepath": "docs/alpha.txt"', "not valid JSON", id="truncated-json"),
    pytest.param("read_file", '["docs/alpha.txt"]', "must be a JSON object", id="json-not-object"),
    pytest.param("read_file", '{"path": "docs/alpha.txt"}', "Bad arguments", id="missing-required-arg"),
]


@pytest.mark.parametrize("name, raw_args, expected_error", BAD_CALLS)
def test_bad_tool_call_returns_an_error_and_the_session_continues(
    workspace, history, name, raw_args, expected_error
):
    client = FakeClient([
        wants_tools(tool_call(name, raw_args, "bad")),
        # The model reads the error and corrects itself on the next pass.
        wants_tools(tool_call("read_file", {"filepath": "docs/alpha.txt"}, "good")),
        answers("It mentions Kafka."),
        # Second user turn on the same history. The old loop raised on the bad
        # call, left it unanswered, and the API rejected every turn after it.
        answers("Still here."),
    ])

    assert agent.run_assistant("What does alpha say?", history, client=client) == "It mentions Kafka."
    error, good = tool_replies(history)
    assert error["success"] is False
    assert expected_error in error["error"]
    assert good["success"] is True
    assert_history_is_valid(history)

    assert agent.run_assistant("Anything else?", history, client=client) == "Still here."
    assert_history_is_valid(client.requests[-1]["messages"])


def test_one_bad_call_does_not_stop_its_siblings_running(workspace, history):
    client = FakeClient([
        wants_tools(
            tool_call("no_such_tool", "{}", "x"),
            tool_call("read_file", {"filepath": "docs/alpha.txt"}, "y"),
        ),
        answers("Done."),
    ])

    agent.run_assistant("go", history, client=client)

    bad, good = tool_replies(history)
    assert bad["success"] is False
    assert good["success"] is True
    assert_history_is_valid(history)


def test_empty_arguments_string_is_treated_as_no_arguments(workspace):
    # Some models send "" rather than "{}" for a call with no arguments. That
    # should reach the tool as a missing-argument error, not a JSON error.
    result = agent.execute_tool_call("list_files", "")
    assert result == {"success": False, "error": result["error"]}
    assert "Bad arguments" in result["error"]


def test_sandbox_refusal_still_reaches_the_model_through_the_loop(workspace, history):
    client = FakeClient([
        wants_tools(tool_call("write_file", {"filepath": "../escaped.txt", "content": "x"}, "w")),
        answers("I can't write there."),
    ])

    agent.run_assistant("write outside", history, client=client)

    (refusal,) = tool_replies(history)
    assert refusal["success"] is False
    assert "outside the permitted workspace" in refusal["error"]
    assert not (workspace.parent / "escaped.txt").exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Iteration cap and API errors
# ─────────────────────────────────────────────────────────────────────────────
def test_iteration_cap_stops_a_model_that_never_answers(workspace, history):
    client = LoopingClient()

    reply = agent.run_assistant("loop forever", history, client=client)

    assert len(client.requests) == agent.MAX_ITERATIONS
    assert "without a final answer" in reply
    # Stopping at the cap must still leave a history the next turn can use.
    assert_history_is_valid(history)


def test_api_error_on_a_tools_request_falls_back_to_a_text_answer(workspace, history):
    client = FakeClient([RuntimeError("400 tool_use_failed"), answers("Plain answer.")])

    assert agent.run_assistant("hi", history, client=client) == "Plain answer."
    assert "tools" not in client.requests[1]
    assert history[-1] == {"role": "assistant", "content": "Plain answer."}


def test_api_error_on_both_attempts_is_reported(workspace, history):
    client = FakeClient([RuntimeError("boom"), RuntimeError("rate limited")])

    assert agent.run_assistant("hi", history, client=client) == "Error: rate limited"
