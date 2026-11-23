"""
The agent as a LangGraph graph: same tools, same sandbox, plus two things the
hand-written loop in llm_file_assistant.py cannot do.

    python graph_agent.py                  # new conversation on thread "default"
    python graph_agent.py --thread alice   # pick a conversation; rerun later to continue it

WHY LANGGRAPH, AND FOR EXACTLY TWO THINGS
─────────────────────────────────────────
1. A human approval pause before write_file (`interrupt`).
2. Memory that survives a restart (a Postgres checkpointer).

Everything else is deliberately unchanged: the tool schemas, the system prompt,
the iteration cap and `execute_tool_call` (and so the sandbox in fs_tools.py)
are imported from llm_file_assistant.py, not rewritten.

CONCEPTS
────────
Graph       The loop, drawn as nodes and edges. `agent` calls the model; if the
            reply asks for tools, the edge goes to `tools`, which runs them and
            goes back to `agent`; otherwise the graph ends. Same loop as before.

State       What flows between nodes: here, the list of messages. Each node
            returns only what it adds, and LangGraph appends it.

Checkpoint  After every node, LangGraph saves the state under a `thread_id`.
            With PostgresSaver those snapshots live in Postgres tables, so a
            new process with the same thread_id picks up exactly where the last
            one stopped, including a pending approval.

interrupt   `interrupt(payload)` inside a node stops the run, saves a
            checkpoint, and hands `payload` to the caller. The caller later
            resumes with `Command(resume=value)`, and `interrupt` returns that
            value. Gotcha: resuming re-runs the interrupted node FROM THE TOP.
            That is why the tools node asks for every approval before running
            any tool: otherwise a read before the pause would run twice, and a
            write approved earlier in the same step would be written twice.
"""

import argparse
import json
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from llm_file_assistant import MAX_ITERATIONS, MODEL, SYSTEM_PROMPT, TOOLS, execute_tool_call

NEEDS_APPROVAL = {"write_file"}
DECLINED = {"success": False, "error": "The user declined this write. Nothing was written."}
CAP_MESSAGE = f"Stopped after {MAX_ITERATIONS} model calls without a final answer."


def model_calls_this_turn(messages) -> int:
    """
    Model replies since the user last spoke. Counted from the saved messages
    rather than by LangGraph's recursion_limit, which starts again every time a
    paused run is resumed, so each approval used to grant a fresh allowance.
    """
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        count += isinstance(m, AIMessage)
    return count


def build_graph(model, checkpointer=None):
    """
    `model` is anything with .invoke(messages) -> AIMessage that already has the
    tools bound: the real ChatGroq in production, a scripted fake in the tests.
    """

    def agent(state: MessagesState):
        # The system prompt is added per call rather than stored, so checkpoints
        # hold only the conversation and a prompt change applies to old threads.
        return {"messages": [model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    def tools(state: MessagesState):
        last = state["messages"][-1]

        # 1. Ask first. Each interrupt() pauses the run; on resume this node
        #    starts again from here, and interrupts already answered return
        #    their stored answers immediately. No side effects happen above
        #    this line, so replaying it is harmless.
        approved = {}
        for call in last.tool_calls:
            if call["name"] in NEEDS_APPROVAL:
                decision = interrupt({"tool": call["name"], "args": call["args"], "id": call["id"]})
                approved[call["id"]] = isinstance(decision, dict) and decision.get("approved") is True

        # 2. Then act, once.
        replies = []
        for call in last.tool_calls:
            if approved.get(call["id"], True):
                result = execute_tool_call(call["name"], json.dumps(call["args"]))
            else:
                result = DECLINED
            replies.append(ToolMessage(json.dumps(result, default=str), tool_call_id=call["id"]))

        # Tool calls whose arguments were not valid JSON land in
        # invalid_tool_calls, not tool_calls. They still need a reply, or the
        # API rejects the conversation from then on: the same bug the
        # hand-written loop had, in a new place.
        for bad in getattr(last, "invalid_tool_calls", []) or []:
            replies.append(ToolMessage(
                json.dumps({"success": False, "error": f"Arguments for {bad.get('name')} are not valid JSON."}),
                tool_call_id=bad["id"],
            ))
        return {"messages": replies}

    def capped(state: MessagesState):
        return {"messages": [AIMessage(CAP_MESSAGE)]}

    def route(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if last.tool_calls or getattr(last, "invalid_tool_calls", None) else END

    def after_tools(state: MessagesState):
        # Checked only after tools have replied, so stopping here never leaves a
        # tool call unanswered.
        return "capped" if model_calls_this_turn(state["messages"]) >= MAX_ITERATIONS else "agent"

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("capped", capped)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_conditional_edges("tools", after_tools, ["agent", "capped"])
    g.add_edge("capped", END)
    return g.compile(checkpointer=checkpointer)


def run_config(thread_id: str) -> dict:
    # Not the cap: model_calls_this_turn is. This is a backstop set above it
    # (a model and a tool step per call, plus the capped step), because hitting
    # it can stop the run between a tool call and its reply.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * MAX_ITERATIONS + 5}


def step(graph, thread_id: str, user_input=None, resume=None) -> dict:
    """
    Advance one turn: start it with `user_input`, or continue a paused one with
    `resume`. Returns {"pending": [approval requests]} or {"answer": text}.
    """
    payload = Command(resume=resume) if resume is not None else {"messages": [HumanMessage(user_input)]}
    result = graph.invoke(payload, run_config(thread_id))
    if result.get("__interrupt__"):
        return {"pending": [i.value for i in result["__interrupt__"]]}
    last = result["messages"][-1]
    return {"answer": last.content if isinstance(last, AIMessage) else str(last.content)}


def pending_approvals(graph, thread_id: str) -> list:
    """Approval requests left waiting by an earlier run of this thread, if any."""
    return [i.value for t in graph.get_state(run_config(thread_id)).tasks for i in t.interrupts]


# ─────────────────────────────────────────────────────────────────────────────
#  Production wiring: Groq model + Postgres checkpointer
# ─────────────────────────────────────────────────────────────────────────────
def groq_model():
    import os

    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    return ChatGroq(model=MODEL, temperature=0).bind_tools(TOOLS)


def postgres_checkpointer(url: str = None):
    """A PostgresSaver on its own connection, with its tables created if missing."""
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    import pg_store

    conn = psycopg.connect(url or pg_store.database_url(), autocommit=True,
                           prepare_threshold=0, row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


def ask_approval(request, ask=input) -> dict:
    args = request["args"]
    content = str(args.get("content", ""))
    preview = content if len(content) <= 400 else content[:400] + f"... ({len(content)} characters)"
    print(f"\n  The assistant wants to write {args.get('filepath')!r}:\n")
    print("    " + preview.replace("\n", "\n    "))
    reply = ask("\n  Allow this write? [y/N] ").strip().lower()
    return {"approved": reply in ("y", "yes")}


def main(argv=None):
    p = argparse.ArgumentParser(description="File assistant with write approval and persistent memory.")
    p.add_argument("--thread", default="default", help="conversation id; reuse it to continue after a restart")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    graph = build_graph(groq_model(), postgres_checkpointer())
    history = graph.get_state(run_config(args.thread)).values.get("messages", [])
    print(f"=== LLM File System Assistant (LangGraph) — thread {args.thread!r}"
          + (f", resuming {len(history)} messages" if history else "") + " ===")
    print("Type 'quit' to exit.\n")

    pending = pending_approvals(graph, args.thread)
    if pending:
        print("A write from your last session is still waiting for approval.")
    while True:
        if pending:
            outcome = step(graph, args.thread, resume=ask_approval(pending[0]))
        else:
            query = input("You: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            outcome = step(graph, args.thread, user_input=query)
        pending = outcome.get("pending", [])
        if "answer" in outcome:
            print(f"\nAssistant: {outcome['answer']}\n")


if __name__ == "__main__":
    main()
