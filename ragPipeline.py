from dotenv import load_dotenv
from embeddingVoyageai import generate_embedding, chunk_by_section

load_dotenv()
import anthropic
import numpy as np
import json
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically


class VectorStore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, query_embedding, k=2):
        query = np.array(query_embedding)

        results = []

        for doc in self.documents:
            embedding = np.array(doc["embedding"])

            # Cosine distance
            similarity = np.dot(query, embedding) / (
                np.linalg.norm(query) * np.linalg.norm(embedding)
            )

            distance = 1 - similarity

            results.append((doc, distance))

        # Lower distance = more similar
        results.sort(key=lambda x: x[1])

        return results[:k]
    
with open("embeddings.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

store = VectorStore(documents)

def add_user_message(messages, text):
    # Logs what the HUMAN said, tagged with role "user"
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    # Logs what CLAUDE said, tagged with role "assistant"
    # Without this, Claude's own replies never get saved into the
    # conversation, so the next call won't know what it said before
    messages.append({"role": "assistant", "content": text})

def chat(messages):
    # Get the user's latest question
    user_query = messages[-1]["content"]
    # 1. Embed the user's question
    userembedding = generate_embedding(user_query)
    # 2. Search for relevant knowledge-base chunks
    results = store.search(userembedding, 2)
    # 3. Build context from retrieved documents
    context = "\n\n---\n\n".join(
        doc["content"] for doc, distance in results
    )
     # 4. Add the retrieved context to the user's message
    augmented_prompt = f"""
Use the following knowledge base context to answer the question.

Context:
{context}

Question:
{user_query}

If the answer is not found in the context, say you don't know.
"""

    # 5. Send augmented prompt to Claude
    params = {                     
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "messages":[
                {
                    "role": "user",
                    "content": augmented_prompt,
                }
            ],
        }
    message = client.messages.create(**params)

     # Debugging / inspection
    for doc, distance in results:
        print(distance, "\n", doc["content"][0:200], "\n")
    print(f"Input tokens: {message.usage.input_tokens}") 
    print(f"Output tokens: {message.usage.output_tokens}")
    return message.content[0].text #use this to return only the text, no headers


messages = []
add_user_message(messages, "What is Kyle's Biggest Cymbal?")
answer = chat(messages)

answer
