import json
import os
from dotenv import load_dotenv
from groq import Groq
import fs_tools

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
def run_assistant(user_query: str, messages: list = None) -> str:
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": user_query})

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            # If tool call fails, retry without tools to get a text response
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e2:
                return f"Error: {e2}"

        msg = response.choices[0].message

        if msg.tool_calls:
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
                fn_args = json.loads(tool_call.function.arguments)

                print(f"  [Tool Call] {fn_name}({fn_args})")
                result = TOOL_MAP[fn_name](**fn_args)
                result_str = json.dumps(result, default=str)
                print(f"  [Result] {result_str[:200]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

            # Force a text response after tool results (no more tool calls)
            try:
                follow_up = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                )
                result_msg = follow_up.choices[0].message.content
                messages.append({"role": "assistant", "content": result_msg})
                return result_msg
            except Exception:
                return result_str

        else:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content

    return "Max iterations reached."


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