from dotenv import load_dotenv
load_dotenv()

import os
import anthropic

BASE_DIR = os.path.dirname(os.path.abspath("toolTextEditor.py"))
target_path = os.path.join(BASE_DIR, "toolUseMulti.py")

client = anthropic.Anthropic()

MODEL = "claude-sonnet-5"


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def text_from_message(message):
    return "\n".join(b.text for b in message.content if b.type == "text")


def get_text_editor_tool():
    return {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }

def resolve_safe_path(path):
    if not path:
        path = "."
    candidate = os.path.abspath(os.path.join(BASE_DIR, path.lstrip("/\\")))
    if not (candidate == BASE_DIR or candidate.startswith(BASE_DIR + os.sep)):
        raise PermissionError(f"Access outside the project directory is not allowed: {path}")
    return candidate

def run_text_editor_tool(tool_use):
    """Actually perform the file operation Claude requested."""
    command = tool_use.input.get("command")

    try:
        path = resolve_safe_path(tool_use.input.get("path"))
    except PermissionError as e:
        return {"content": str(e), "is_error": True}

    try:
        if command == "view":
            if os.path.isdir(path):
                return "\n".join(os.listdir(path))
            with open(path) as f:
                lines = f.readlines()
            numbered = "".join(f"{i}\t{l}" for i, l in enumerate(lines, 1))
            return f"Here's the result of running `cat -n` on {path}:\n{numbered}"

        elif command == "create":
            with open(path, "w") as f:
                f.write(tool_use.input.get("file_text", ""))
            return f"File created at {path}"

        elif command == "str_replace":
            with open(path) as f:
                content = f.read()
            old_str = tool_use.input["old_str"]
            new_str = tool_use.input.get("new_str", "")
            if content.count(old_str) != 1:
                return {"content": f"old_str must match exactly once in {path}", "is_error": True}
            with open(path, "w") as f:
                f.write(content.replace(old_str, new_str))
            return "Successfully replaced text at exactly one location."

        elif command == "insert":
            with open(path) as f:
                lines = f.readlines()
            insert_line = tool_use.input["insert_line"]
            new_str = tool_use.input.get("new_str", "")
            lines.insert(insert_line, new_str if new_str.endswith("\n") else new_str + "\n")
            with open(path, "w") as f:
                f.writelines(lines)
            return f"Successfully inserted text after line {insert_line}"

        return {"content": f"Unknown command: {command}", "is_error": True}

    except FileNotFoundError:
        return {"content": f"File not found: {path}", "is_error": True}
    except Exception as e:
        return {"content": f"Error: {e}", "is_error": True}

def run_conversation(messages):
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
            tools=[get_text_editor_tool()],
        )
        add_assistant_message(messages, response.content)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            return response

        tool_results = []
        for tool_use in tool_uses:
            print("\nSelected content block:", tool_use)
            result = run_text_editor_tool(tool_use)
            print("\nTool result:", result)

            if isinstance(result, dict):
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result["content"],
                    "is_error": result.get("is_error", False),
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

        add_user_message(messages, tool_results)


messages = []
add_user_message(messages, f"Open the {target_path} file and summarize what it does")

final_response = run_conversation(messages)
print("\nClaude's final response:")
print(text_from_message(final_response))