from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import os
import re

application = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def create_prompt(retrieved_content, user_question):
    prompt = f"Context:\n{retrieved_content}\n\nQuestion: {user_question}\n\nAnswer:"
    #print(f"Retrieved content: {retrieved_content}\n--------------\n")
    return prompt

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant helping with questions about a specific mission. Use the provided context to answer accurately. Your tone should be somewhat robotic, stilted, and serious. Act like this is serious business and not an actual game. Keep the answers brief, since this is a mission-critical situation where time is of the essence."},
            {"role": "user", "content": prompt}
        ]
    )
    response_str = response.choices[0].message.content
    return response_str 

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

def query_database(query: str, k: int = 1):
    matching_docs = vector_db.similarity_search(query, k=k)
    return matching_docs if matching_docs else []

def mission_qa(user_question):
    relevant_docs = query_database(user_question, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = create_prompt(context, user_question)
    print("Prompt: ", prompt)
    response = chat_with_gpt(prompt)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = mission_qa(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
