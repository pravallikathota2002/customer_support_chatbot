"""Reads the previous turn of a session from the database."""

import json
import re
from typing import Any

from agno.db.base import SessionType

from chatbot.services.database import db

MAX_TEXT_CHARS = 300

NO_HISTORY = {"question": "none", "answer": "none", "agent": "none", "follow_ups": []}


def _parse_content(content: Any) -> dict:
    """The agent reply is normally a dict; fall back to parsing JSON text."""
    if isinstance(content, dict):
        return content
    match = re.search(r"\{.*\}", str(content or ""), re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"response": str(content or ""), "follow_up_questions": []}


def _agent_label(run) -> str:
    """'General Assistant' -> general_agent, 'Knowledge Assistant' -> knowledge_agent."""
    name = f"{run.agent_name or ''} {run.agent_id or ''}".lower()
    if "troubleshoot" in name:
        return "troubleshooting_agent"
    if "technician" in name :
        return "technician_agent"
    if "knowledge" in name:
        return "knowledge_agent"
    if "general" in name:
        return "general_agent"
    return "none"


def get_previous(session_id: str) -> dict:
    """Return the last question, answer, agent and follow-up questions."""
    try:
        session = db.get_session(session_id=session_id, session_type=SessionType.AGENT)
    except Exception as error:
        print(f"[history] could not read session: {error}")
        return NO_HISTORY

    if session is None or not session.runs:
        return NO_HISTORY

    last_run = session.runs[-1]
    reply = _parse_content(last_run.content)

    return {
        "question": str(last_run.input.input_content)[:MAX_TEXT_CHARS],
        "answer": str(reply.get("response", ""))[:MAX_TEXT_CHARS] or "none",
        "agent": _agent_label(last_run),
        "follow_ups": list(reply.get("follow_up_questions", [])),
    }