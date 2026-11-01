import uuid
from typing import List, Optional
import os 


from fastapi import FastAPI, HTTPException,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.chat_service import chat
from fastapi.responses import JSONResponse, Response



router = APIRouter(prefix="/vontier/agents")

# from frontend we are getting sessionid and message
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # send the same id for the whole conversation


# need to add tools used also
class ChatResponse(BaseModel):
    session_id: str
    agent_used: str
    response: str
    followups: List[str]


#need to work more on tools -- it was not creating tickets giving dummy data 

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = chat(req.message, session_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat failed: {error}")

    return ChatResponse(session_id=session_id, **result)

