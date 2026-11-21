"""
Tests for graph_agent.py: approval before write_file, and memory that survives
a restart. No network: a scripted fake model replaces Groq.

The approval tests use LangGraph's in-memory checkpointer, which behaves the
same as Postgres for pausing and resuming. The restart tests use the real
PostgresSaver against a throwaway schema, and simulate a restart by throwing
away the graph, the checkpointer and its connection, then building new ones.
"""

import copy
import json
import uuid

import pytest

pytest.importorskip("langgraph")
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import graph_agent  # noqa: E402


class FakeModel:
    """Replays scripted AIMessages; records the messages each call received."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if not self.script:
            raise AssertionError("The graph made more model calls than the test scripted.")
        return self.script.pop(0)


class LoopingModel(FakeModel):
    def __init__(self):
        super().__init__([])

    def invoke(self, messages):
        self.calls.append(None)
        return wants(call("list_files", {"directory": "docs"}, f"c{len(self.calls)}"))


def call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def wants(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def says(text):
    return AIMessage(content=text)


def write(path, content="report", call_id="w1"):
    return call("write_file", {"filepath": path, "content": content}, call_id)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_ROOT", str(tmp_path))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "alpha.txt").write_text("Python developer, Kafka\n", encoding="utf-8")
    return tmp_path


def tool_results(graph, thread):
    msgs = graph.get_state(graph_agent.run_config(thread)).values["messages"]
    return [json.loads(m.content) for m in msgs if m.type == "tool"]


# ─────────────────────────────────────────────────────────────────────────────
#  Approval before write_file
# ─────────────────────────────────────────────────────────────────────────────
def test_write_pauses_for_approval_and_nothing_is_written_yet(workspace):
    model = FakeModel([wants(write("out/a.md", "hello")), says("Done.")])
    graph = graph_agent.build_graph(model, InMemorySaver())

    out = graph_agent.step(graph, "t", user_input="write it")

    assert out["pending"] == [{"tool": "write_file", "args": {"filepath": "out/a.md", "content": "hello"}, "id": "w1"}]
    assert not (workspace / "out" / "a.md").exists()
    assert len(model.calls) == 1


def test_approved_write_happens_exactly_once(workspace):
    model = FakeModel([wants(write("out/a.md", "hello")), says("Written.")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    graph_agent.step(graph, "t", user_input="write it")

    out = graph_agent.step(graph, "t", resume={"approved": True})

    assert out == {"answer": "Written."}
    assert (workspace / "out" / "a.md").read_text(encoding="utf-8") == "hello"
    assert tool_results(graph, "t")[0]["success"] is True


def test_declined_write_writes_nothing_and_tells_the_model(workspace):
    model = FakeModel([wants(write("out/a.md")), says("OK, I won't.")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    graph_agent.step(graph, "t", user_input="write it")

    assert graph_agent.step(graph, "t", resume={"approved": False}) == {"answer": "OK, I won't."}
    assert not (workspace / "out").exists()
    assert tool_results(graph, "t") == [graph_agent.DECLINED]
    # The model saw the refusal as the tool's result.
    assert "declined" in model.calls[-1][-1].content


@pytest.mark.parametrize("decision", [{}, {"approved": "yes"}, {"approved": 1}, "y", False])
def test_anything_but_an_explicit_true_counts_as_declined(workspace, decision):
    model = FakeModel([wants(write("out/a.md")), says("ok")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    graph_agent.step(graph, "t", user_input="write it")
    graph_agent.step(graph, "t", resume=decision)
    assert not (workspace / "out").exists()


def test_reads_need_no_approval(workspace):
    model = FakeModel([wants(call("read_file", {"filepath": "docs/alpha.txt"}, "r1")), says("It says Kafka.")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    assert graph_agent.step(graph, "t", user_input="read") == {"answer": "It says Kafka."}


def test_read_alongside_a_write_runs_once_not_again_on_resume(workspace, monkeypatch):
    """
    Resuming re-runs the paused node from the top. If tools ran before the
    approval was asked, the read would run twice. Count calls to make sure.
    """
    calls = []
    real = graph_agent.execute_tool_call
    monkeypatch.setattr(graph_agent, "execute_tool_call", lambda n, a: calls.append(n) or real(n, a))
    model = FakeModel([
        wants(call("read_file", {"filepath": "docs/alpha.txt"}, "r1"), write("out/a.md", "x", "w1")),
        says("Done."),
    ])
    graph = graph_agent.build_graph(model, InMemorySaver())

    graph_agent.step(graph, "t", user_input="read then write")
    assert calls == []  # nothing runs until every approval is in
    graph_agent.step(graph, "t", resume={"approved": True})
    assert calls == ["read_file", "write_file"]


def test_two_writes_in_one_step_are_approved_separately(workspace):
    model = FakeModel([wants(write("out/a.md", "A", "w1"), write("out/b.md", "B", "w2")), says("Done.")])
    graph = graph_agent.build_graph(model, InMemorySaver())

    assert graph_agent.step(graph, "t", user_input="two")["pending"][0]["id"] == "w1"
    assert graph_agent.step(graph, "t", resume={"approved": True})["pending"][0]["id"] == "w2"
    assert graph_agent.step(graph, "t", resume={"approved": False}) == {"answer": "Done."}

    assert (workspace / "out" / "a.md").read_text(encoding="utf-8") == "A"
    assert not (workspace / "out" / "b.md").exists()


def test_approval_does_not_bypass_the_sandbox(workspace):
    model = FakeModel([wants(write("../escaped.md")), says("Blocked.")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    graph_agent.step(graph, "t", user_input="write outside")

    graph_agent.step(graph, "t", resume={"approved": True})

    assert not (workspace.parent / "escaped.md").exists()
    assert "outside the permitted workspace" in tool_results(graph, "t")[0]["error"]


# ─────────────────────────────────────────────────────────────────────────────
#  Bad calls and the iteration cap (carried over from the hand-written loop)
# ─────────────────────────────────────────────────────────────────────────────
def test_unknown_tool_comes_back_as_an_error_and_the_conversation_continues(workspace):
    model = FakeModel([wants(call("delete_everything", {}, "x")), says("Sorry."), says("Still here.")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    assert graph_agent.step(graph, "t", user_input="go") == {"answer": "Sorry."}
    assert "Unknown tool" in tool_results(graph, "t")[0]["error"]
    assert graph_agent.step(graph, "t", user_input="again") == {"answer": "Still here."}


def test_tool_call_with_unparseable_arguments_still_gets_a_reply(workspace):
    bad = AIMessage(content="", invalid_tool_calls=[
        {"name": "read_file", "args": '{"filepath": ', "id": "bad1", "error": "json", "type": "invalid_tool_call"}])
    model = FakeModel([bad, says("Let me retry.")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    assert graph_agent.step(graph, "t", user_input="go") == {"answer": "Let me retry."}
    msgs = graph.get_state(graph_agent.run_config("t")).values["messages"]
    assert [m.tool_call_id for m in msgs if m.type == "tool"] == ["bad1"]


def test_iteration_cap_stops_a_model_that_never_answers(workspace):
    model = LoopingModel()
    graph = graph_agent.build_graph(model, InMemorySaver())
    assert graph_agent.step(graph, "t", user_input="loop") == {"answer": graph_agent.CAP_MESSAGE}
    assert len(model.calls) == graph_agent.MAX_ITERATIONS


def test_threads_are_separate_conversations(workspace):
    model = FakeModel([says("one"), says("two")])
    graph = graph_agent.build_graph(model, InMemorySaver())
    graph_agent.step(graph, "a", user_input="hi from a")
    graph_agent.step(graph, "b", user_input="hi from b")
    assert [m.content for m in model.calls[1][1:]] == ["hi from b"]


# ─────────────────────────────────────────────────────────────────────────────
#  Memory across restarts: the real Postgres checkpointer
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def pg_url():
    pytest.importorskip("langgraph.checkpoint.postgres")
    psycopg = pytest.importorskip("psycopg")
    import pg_store

    try:
        base = pg_store.database_url()
        admin = psycopg.connect(base, autocommit=True, connect_timeout=5)
    except Exception as e:
        pytest.skip(f"no Postgres available: {e}")
    schema = f"test_ckpt_{uuid.uuid4().hex[:10]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in base else "?"
    yield f"{base}{sep}options=-csearch_path%3D{schema}"
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def restart(model, url):
    """A fresh process, as far as the graph can tell: new saver, new connection."""
    saver = graph_agent.postgres_checkpointer(url)
    return graph_agent.build_graph(model, saver), saver


def test_conversation_survives_a_restart(workspace, pg_url):
    model = FakeModel([says("Noted: you like Kafka."), says("You said Kafka.")])
    graph, saver = restart(model, pg_url)
    graph_agent.step(graph, "alice", user_input="Remember that I like Kafka.")
    saver.conn.close()
    del graph, saver

    graph, saver = restart(model, pg_url)
    assert graph_agent.step(graph, "alice", user_input="What do I like?") == {"answer": "You said Kafka."}
    # The second process's model call saw the first process's turn.
    assert [m.content for m in model.calls[1][1:]] == [
        "Remember that I like Kafka.", "Noted: you like Kafka.", "What do I like?"]
    saver.conn.close()


def test_pending_approval_survives_a_restart(workspace, pg_url):
    model = FakeModel([wants(write("out/a.md", "hello")), says("Written.")])
    graph, saver = restart(model, pg_url)
    assert "pending" in graph_agent.step(graph, "bob", user_input="write it")
    saver.conn.close()

    graph, saver = restart(model, pg_url)
    assert graph_agent.pending_approvals(graph, "bob")[0]["args"]["filepath"] == "out/a.md"
    assert not (workspace / "out").exists()
    assert graph_agent.step(graph, "bob", resume={"approved": True}) == {"answer": "Written."}
    assert (workspace / "out" / "a.md").read_text(encoding="utf-8") == "hello"
    saver.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  The terminal prompt
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("reply, approved", [("y", True), ("YES", True), ("", False), ("n", False), ("sure", False)])
def test_ask_approval_defaults_to_no(reply, approved, capsys):
    req = {"tool": "write_file", "args": {"filepath": "out/a.md", "content": "x" * 1000}, "id": "w1"}
    assert graph_agent.ask_approval(req, ask=lambda _: reply) == {"approved": approved}
    assert "1000 characters" in capsys.readouterr().out
