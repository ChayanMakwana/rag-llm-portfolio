import os
from openai import OpenAI
from retriever import retrieve_chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_rag_question(query):
    relevant_chunks = retrieve_chunks(query)
    context = "\n\n".join(relevant_chunks)

    system_prompt = (
        "You are an AI assistant that answers questions using the provided context. "
        "If the context does not contain an answer, say you don't know."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() in {"exit", "quit"}:
            break
        answer = ask_rag_question(q)
        print("\nAnswer:\n", answer)

