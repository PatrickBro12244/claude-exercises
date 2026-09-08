import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from dotenv import load_dotenv
    load_dotenv()

    import datetime
    import anthropic
    from anthropic.types import ToolParam

    client = anthropic.Anthropic()

    # Program for practicing tool use using Claude API

    def add_user_message(messages, text):
        messages.append({
            "role": "user",
            "content": text
        })

    def add_assistant_message(messages, content):
        messages.append({
            "role": "assistant",
            "content": content
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
                    "description": (
                        "A Python strftime format string."
                    ),
                    "default": "%Y-%m-%d %H:%M:%S"
                }
            },
            "required": []
        }
    })

    def chat(messages, system=None):
        params = {
            "model": "claude-haiku-4-5",
            "max_tokens": 1000,
            "messages": messages,
            "tools": [get_current_datetime_schema]
        }

        if system:
            params["system"] = system

        # First request
        response = client.messages.create(**params)

        print(f"\nInput Tokens: {response.usage.input_tokens}")
        print(f"Output Tokens: {response.usage.output_tokens}")

        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":

            # Add Claude's response to conversation
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []

            for content in response.content:

                if content.type == "tool_use":

                    if content.name == "get_current_datetime":

                        result = get_current_datetime(
                            **content.input
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result
                        })

            # Send the tool result back to Claude
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Second request
            response = client.messages.create(**params)

            print(f"\nInput Tokens: {response.usage.input_tokens}")
            print(f"Output Tokens: {response.usage.output_tokens}")

        # Return Claude's final response
        for content in response.content:
            if content.type == "text":
                return content.text

        return None

    return add_user_message, chat


@app.cell
def _(add_user_message, chat):

    messages = []

    add_user_message(
        messages,
        "What is the exact current time? "
        "Use the available tool and format it as HH:MM:SS."
    )

    response = chat(messages)

    print(response)
    return


if __name__ == "__main__":
    app.run()
