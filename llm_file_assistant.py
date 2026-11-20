import json
import os
import fs_tools

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Only a bare test environment lacks it; the loop tests need no .env.
    pass

MODEL = "openai/gpt-oss-20b"
MAX_ITERATIONS = 10

_client = None


def get_client():
    # Created (and groq imported) on first use, not at import: Groq() raises
    # without an API key, and the tests import this module without one.
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a resume file (PDF, TXT, DOCX) and extract its text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in a directory, optionally filtered by extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path to list"},
                    "extension": {"type": "string", "description": "File extension filter, e.g. .pdf, .txt"},
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file. Creates directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to write the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search for a keyword in a file. Returns matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to search"},
                    "keyword": {"type": "string", "description": "Keyword to search for (case-insensitive)"},
                },
                "required": ["filepath", "keyword"],
            },
        },
    },
]

TOOL_MAP = {
    "read_file": fs_tools.read_file,
    "list_files": fs_tools.list_files,
    "write_file": fs_tools.write_file,
    "search_in_file": fs_tools.search_in_file,
}

SYSTEM_PROMPT = """You are a helpful file system assistant specializing in resume management.
You can read, list, search, and write files using the provided tools.
Always use tools to interact with the file system. Be concise and helpful.
IMPORTANT: The resumes are stored in the 'resumes' folder (relative path). Always use relative paths like 'resumes/' not absolute paths.
When searching multiple files, search them one at a time. Only call ONE tool at a time.
IMPORTANT: Always call list_files first to get actual filenames before reading or searching files. Never guess filenames.
When listing files, do NOT pass an extension filter unless the user specifically asks for a certain file type. Call list_files with only the directory parameter."""


def execute_tool_call(name: str, raw_arguments: str):
    """
    Run one tool call the model asked for. Never raises.

    By the time this runs, the assistant message carrying the tool call is already
    in the history, and the API rejects every later request in which a tool call
    has no matching tool reply. So a bad call (unknown name, malformed JSON, wrong
    arguments) must come back as a result the model can read and correct. An
    exception here would leave the call unanswered and break the whole session.
    """
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {
            "success": False,
            "error": f"Unknown tool: {name!r}. Available tools: {', '.join(TOOL_MAP)}.",
        }

    try:
        args = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Arguments for {name} are not valid JSON: {e}"}
    if not isinstance(args, dict):
        return {"success": False, "error": f"Arguments for {name} must be a JSON object."}

    try:
        return fn(**args)
    except TypeError as e:
        # Missing required argument, e.g. read_file() with no filepath.
        return {"success": False, "error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"success": False, "error": f"{name} failed: {e}"}


def run_assistant(user_query: str, messages: list = None, client=None) -> str:
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if client is None:
        client = get_client()

    messages.append({"role": "user", "content": user_query})

    for _ in range(MAX_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception:
            # Groq answers 400 (tool_use_failed) when the model emits a malformed
            # tool call. Retry without tools to get a plain text answer.
            try:
                response = client.chat.completions.create(model=MODEL, messages=messages)
            except Exception as e2:
                return f"Error: {e2}"
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            return content

        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in msg.tool_calls
            ],
        })

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            print(f"  [Tool Call] {fn_name}({tool_call.function.arguments})")
            result = execute_tool_call(fn_name, tool_call.function.arguments)
            result_str = json.dumps(result, default=str)
            print(f"  [Result] {result_str[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })

        # No return here. The results go back to the model on the next pass, and
        # it decides whether it needs another tool or can answer. That is what
        # lets it chain list_files -> read_file -> write_file.

    return f"Stopped after {MAX_ITERATIONS} model calls without a final answer."


def main():
    print("=== LLM File System Assistant ===")
    print("Type 'quit' to exit.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        try:
            response = run_assistant(query, messages)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()