from dotenv import load_dotenv
load_dotenv()

import anthropic
import datetime
from anthropic.types import ToolParam

client = anthropic.Anthropic()

MODEL = "claude-sonnet-5"


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def text_from_message(message):
    return "\n".join(b.text for b in message.content if b.type == "text")


def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")

    return datetime.datetime.now().strftime(date_format)


get_current_datetime_schema = ToolParam({
        "name": "get_current_datetime",
        "description": (
            "Returns the current date and time formatted "
            "according to the specified format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": (
                        "A Python strftime format string."
                    ),
                    "default": "%Y-%m-%d %H:%M:%S"
                }
            },
            "required": []
        }
    })


def run_conversation(messages):
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
            tools=[get_current_datetime_schema],
        )

        add_assistant_message(messages, response.content)

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = get_current_datetime(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            add_user_message(messages, tool_results)
            continue

        return response


messages = []
add_user_message(
    messages,
    "What date is it today"
)

final_response = run_conversation(messages)
print("\nClaude's final response:")
print(text_from_message(final_response))