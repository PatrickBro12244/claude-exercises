from dotenv import load_dotenv
load_dotenv()

import datetime
import anthropic
from anthropic.types import ToolParam

client = anthropic.Anthropic()


# Add user message
def add_user_message(messages, text):
    messages.append({
        "role": "user",
        "content": text
    })


# Get Date and Time tool
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")

    return datetime.datetime.now().strftime(date_format)


# Tool schema given to Claude
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
                "description": "A Python strftime format string.",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        },
        "required": []
    }
})


def chat(messages):

    # First API call
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,
        tools=[get_current_datetime_schema]
    )

    print("\nClaude's first response:")
    print(response.content)

    if response.stop_reason == "tool_use":

        # Add Claude's tool request to the conversation
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # Find the ToolUseBlock
        tool_use = response.content[1]

        print("\nSelected content block:")
        print(tool_use)

        # Execute the tool
        result = get_current_datetime(**tool_use.input)

        print("\nTool result:")
        print(result)

        # Send tool result back to Claude
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result
            }]
        })

        # Second API call
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            messages=messages,
            tools=[get_current_datetime_schema]
        )

    # Return Claude's final response
    return response.content[0].text


# -------------------------
# Test the tool
# -------------------------

messages = []

add_user_message(
    messages,
    "What is the exact current date and time? "
    "First briefly explain that you will check the current date and time, "
    "then use the get_current_datetime tool to get the exact value."
)

response = chat(messages)

print("\nClaude's final response:")
print(response)