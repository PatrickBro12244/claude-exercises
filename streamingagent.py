from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

def add_user_message(messages, text):
    # Logs what the HUMAN said, tagged with role "user"
    messages.append({"role": "user", "content": text})

messages = []
add_user_message(messages, "Write a 1 sentence description of a fake database")
with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=500,
    messages=messages,
) as stream:
    for text in stream.text_stream:
        print(text,end="")

    final_message=stream.get_final_message() #for printing token use
    print(f"\n Input Tokens:{final_message.usage.input_tokens}")
    print(f"\n Output Tokens:{final_message.usage.output_tokens}")
