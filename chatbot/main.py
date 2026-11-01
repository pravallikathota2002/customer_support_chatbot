from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add the project root (vontier_chatbot) to the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from chatbot.api.chat_router import router

app = FastAPI(title="Vontier Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}