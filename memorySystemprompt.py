from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

#Program for stateful claude with system prompt

def add_user_message(messages, text):
    # Logs what the HUMAN said, tagged with role "user"
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    # Logs what CLAUDE said, tagged with role "assistant"
    # Without this, Claude's own replies never get saved into the
    # conversation, so the next call won't know what it said before
    messages.append({"role": "assistant", "content": text})

def chat(messages, system=None):
    params = {                        #need to use this to prevent system=None error
        "model": "claude-sonnet-4-6",
        "max_tokens": 500,
        "messages": messages,
    }
    if system:
        params["system"] = system
    message = client.messages.create(**params)
    print(f"Input tokens: {message.usage.input_tokens}") 
    print(f"Output tokens: {message.usage.output_tokens}")
    return message.content[0].text #use this to return only the text, no headers

system = """
You are a programmer working with an extremely strict boss. He mandates to write python code and functions as short and as efficiently as possible or else he will fire you without pay.
"""
messages = []
add_user_message(messages, "Write a python function that checks a string for duplicate characters")
answer = chat(messages,system)
answer
