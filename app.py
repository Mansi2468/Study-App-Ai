import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_DB_URI")

# MongoDB setup
try:
    client = MongoClient(mongo_uri)
    db = client['chat']
    collection = db['users']
    # Test connection
    client.server_info()
except Exception as e:
    print(f"MongoDB Connection Error: {e}")
    client = None
    db = None
    collection = None

# FastAPI setup
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain if needed
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant."),
    ("human", "{question}")
])

# Use Groq LLaMA 3.1 model (currently supported)
llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
chain = prompt | llm

# Retrieve chat history
def get_chat_history(user_id):
    if collection is None:
        return []
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []
    for chat in chats:
        history.append({"role": chat['role'], "content": chat['message']})
    return history

def format_history(history):
    return "\n".join([f"{h['role']}: {h['content']}" for h in history])

@app.get("/")
def home():
    return {"message": "Welcome to the Groq LLaMA Chat API!"}

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_chat_history(request.user_id)
    formatted_history = format_history(history)

    # Include history in the question if it exists
    if formatted_history:
        full_question = f"Previous conversation:\n{formatted_history}\n\nCurrent question: {request.question}"
    else:
        full_question = request.question

    response = chain.invoke({"question": full_question})

    # Save user and assistant messages
    if collection is not None:
        collection.insert_one({
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": datetime.utcnow()
        })
        collection.insert_one({
            "user_id": request.user_id,
            "role": "assistant",
            "message": response.content,
            "timestamp": datetime.utcnow()
        })

    return {"response": response.content}
