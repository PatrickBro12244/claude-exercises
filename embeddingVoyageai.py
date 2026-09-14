from dotenv import load_dotenv
import voyageai
import re
import json
load_dotenv()
client = voyageai.Client()

def generate_embedding(texts,model="voyage-3-large",input_type="query"):
    result = client.embed(texts, model=model, input_type=input_type)
    return result.embeddings

def chunk_by_section(document_text):
    
    document_text = re.sub(
        r"^# Cymbal Knowledge Base\s*",
        "",
        document_text,
        flags=re.MULTILINE
    )

    # Split before each numbered ## heading
    pattern = r"(?=^## \d+\. )"

    return [
        chunk.strip()
        for chunk in re.split(
            pattern,
            document_text,
            flags=re.MULTILINE
        )
        if chunk.strip()
    ]


with open ("./sampledocument.md", "r") as f:
    text = f.read()

    chunks = chunk_by_section(text)
    print("Number of chunks:", len(chunks))

    embeddings = generate_embedding(chunks)
    print("Number of embeddings:", len(embeddings))
    print("Dimensions of first embedding:", len(embeddings[0]))
    documents = []
   
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        documents.append({
            "id": i,
            "content": chunk,
            "embedding": embedding
        })

    with open('embeddings.json', 'w', encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    