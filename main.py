from ollama import Client
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = chromadb.PersistentClient()
remote_client = Client(host=f"http://localhost:11434")
collection = client.get_or_create_collection(name="articles_demo")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separators=[". "]
)
with open("counter.txt","r") as f:
    count=int(f.read().strip())

print("Reading output.json and generating embeddings...")
with open("output.json", "r",encoding='utf-8') as f:
    json_content=json.load(f)
    for i, article in enumerate(json_content):
        content = article["Body"]
        if i < count:
            continue
        count+=1
        sentences = text_splitter.split_text(content)
        for each_sentence in sentences:
            response = remote_client.embed(model="nomic-embed-text", input=f"search_document: {each_sentence}")
            embedding = response["embeddings"][0]
            collection.add(
                ids=[f"article_{i}"],
                embeddings=[embedding],
                documents=[each_sentence],
                metadatas=[{"title": article["Headline"]}],
            )
print("Database built successfully!")

with open("counter.txt","w") as f:
   f.write(str(count))

while True:
    query=input("How may i help you ? \n")
    if query=="bye":
        break
    query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
    results = collection.query(query_embeddings=[query_embed], n_results=1)
    context='\n'.join(results["documents"][0])
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

    Context: {context}

    Question: {query}

    Answer:"""

    response = remote_client.generate(
            model="qwen3:4b-q4_K_M",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )
    answer = response['response']
    print(answer+'\n')
    print("-"*40)
