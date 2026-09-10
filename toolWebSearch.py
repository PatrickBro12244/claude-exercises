from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-sonnet-5"


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def text_from_message(message):
    return "\n".join(b.text for b in message.content if b.type == "text")


def get_web_search_tool():
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
    }


def run_conversation(messages):
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
            tools=[get_web_search_tool()],
        )

        add_assistant_message(messages, response.content)

        if response.stop_reason == "pause_turn":
            continue

        return response


messages = []
add_user_message(
    messages,
    "What cymbals did Jack Dejhonette play in 1990"
)

final_response = run_conversation(messages)
print("\nClaude's final response:")
print(text_from_message(final_response))

for block in final_response.content:
    if block.type == "text" and block.citations:
        for c in block.citations:
            print(f"  cited: {c.title} — {c.url}")