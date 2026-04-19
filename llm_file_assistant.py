import json
import os
from dotenv import load_dotenv
import openai
import fs_tools

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

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
IMPORTANT: The resumes are stored in the 'resumes' folder (relative path). Always use relative paths like 'resumes/' not absolute paths."""


def run_assistant(user_query: str, messages: list = None) -> str:
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": user_query})

    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        response = openai.ChatCompletion.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg["content"]

        for tool_call in tool_calls:
            fn_name = tool_call["function"]["name"]
            fn_args = json.loads(tool_call["function"]["arguments"])

            print(f"  [Tool Call] {fn_name}({fn_args})")
            result = TOOL_MAP[fn_name](**fn_args)
            result_str = json.dumps(result, default=str)
            print(f"  [Result] {result_str[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result_str,
            })

    return "Max iterations reached. Last tool results were returned above."

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