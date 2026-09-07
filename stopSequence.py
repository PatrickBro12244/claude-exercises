from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

#program for formatting claude ouputs using stop sequences and message prefilling
#message prefilling is deprecated as of 4-6 onwards

def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def chat(messages):
    message = client.messages.create(
        model="claude-sonnet-4-5", 
        max_tokens=500,
        messages=messages,
        stop_sequences=["```"],
    )
    print(f"Input tokens: {message.usage.input_tokens}") 
    print(f"Output tokens: {message.usage.output_tokens}")
    response = message.content[0].text
    return response

# Fix: Ensure the conversation starts and ends with a user message
messages = []
add_user_message(messages, "Generate 3 different sample AWS CLI commands. each should be very short")
add_assistant_message(messages, "```bash")
#assistant prefill is deprecated for claude 4.6 and up
#Below could also work, but for me atp use a system prompt breh
#add_assistant_message(messages, "Here are all 3 AWS CLI commands without any comments or linebreaks```bash")

answer = chat(messages)
print(answer)
