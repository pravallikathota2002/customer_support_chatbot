"""The complete chat flow in one method."""

import json
import re
from typing import Any

from agno.guardrails import PromptInjectionGuardrail

from agno.run import RunStatus

from chatbot.agents.general_agent import general_agent
from chatbot.agents.vontier_knowlege_agent import knowledge_agent
from chatbot.agents.troubleshooting_agent import troubleshooting_agent
from chatbot.agents.technician_agent import technician_agent
from chatbot.services.jev_client import DEFAULT_AGENT, get_agent_for_query

AGENTS = {
    "general_agent": general_agent,
    "knowledge_agent": knowledge_agent,
    "troubleshooting_agent" :troubleshooting_agent,
    "technician_agent": technician_agent
}

FALLBACK_REPLY = {
    "response": "Sorry, I could not answer that. Please try again.",
    "followups": [],
}

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])')
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

BLOCKED_RESPONSE = {
    "response": (
        "I'm sorry, but I can't help with that request.\n\n"
        "- I can't ignore my instructions or share internal or confidential company data.\n"
        "- Protecting customer and company information is a top priority for us.\n\n"
        "I'd be glad to help with fleet management, vehicle tracking, or driver monitoring instead."
    ),
    "followups": [
        "What services are offered by you?",
        "What is fleet management?",
        "How can I track my vehicles?",
    ],
}


def _format_sentences_and_bullets(text: str) -> str:
    """Return only sentences and '- ' bullets, one per line. Nothing is removed."""
    text = text.replace("\r\n", "\n").strip()
    out = []

    for raw in text.split("\n"):
        line = re.sub(r"\*\*|__", "", raw).strip()   # drop bold markers
        line = re.sub(r"^#{1,6}\s+", "", line)        # drop markdown headers
        if not line:
            continue

        if BULLET.match(line):
            # Normalise every bullet style (•, *, 1., 2)) to "- "
            out.append("- " + BULLET.sub("", line, count=1).strip())
        else:
            # Paragraph text: one sentence per line
            out.extend(s.strip() for s in SENTENCE_SPLIT.split(line) if s.strip())

    return "\n".join(out)

def _to_reply(content: Any) -> dict:
    """Convert agent output into a consistent API response."""

    if content is None:
        return dict(FALLBACK_REPLY)

    if hasattr(content, "model_dump"):
        data = content.model_dump()

    elif isinstance(content, dict):
        print("inside the dict=============")

        data = content

    else:
        text = str(content)

        # Parse JSON only if the entire text is valid JSON.
        try:
            data = json.loads(text)
            print(data)
        except (json.JSONDecodeError, TypeError):
            # Plain-text agent response: preserve it completely.
            return {
                "response": text or FALLBACK_REPLY["response"],
                "followups": [],
            }

    if not isinstance(data, dict):
        return {
            "response": str(data),
            "followups": [],
        }

    # Do not split or truncate the agent's answer.
    response = _format_sentences_and_bullets(str(data.get("response") or ""))

    followups = (
        data.get("follow_up_questions")
        or data.get("followups")
        or []
    )

    print("========== COMPLETE AGENT RESPONSE ==========")
    print(repr(response))
    print("Response length:", len(response))

    return {
        "response": response or FALLBACK_REPLY["response"],
        "followups": list(followups)[:3]
        if isinstance(followups, (list, tuple))
        else [],
    }

OFF_TOPIC_RESPONSE = {
    "response": (
        "I'm sorry, but I'm the **Fleet Management Assistant**,\n\n"
        "I'm here to help with:\n"
        "- Fleet and vehicle management\n"
        "- Vehicle performance and maintenance issues\n"
        "- Driver monitoring and device installation\n\n"
        "Please ask me anything related to these topics."
    ),
    "followups": [
        "What services are offered by you?",
        "What is fleet management?",
        "How can I track my vehicles?"
    ]
}


def chat(message: str, session_id: str) -> dict:
    """Pick the agent, run it and return the complete response."""

    try:
        agent_name = get_agent_for_query(message, session_id)["agent"]
    except Exception as error:
        print(f"[chat] routing failed, using {DEFAULT_AGENT}: {error}")
        agent_name = DEFAULT_AGENT

    agent = AGENTS.get(agent_name, knowledge_agent)
    
    print("[chat] Message:", repr(message))
    print("[chat] Session:", session_id)
    print("[chat] Selected agent:", agent_name)
    if agent_name == "off_topic_agent":
        return {
            "agent_used": "off_topic_agent",
            **OFF_TOPIC_RESPONSE
        }

    run = agent.run(message, session_id=session_id,user_id="user_123",metadata={"org_id": "org_456"},)
    print("============================== THIS IS ORIGINAL RESPONSE FROM AGEN===================",run.content)

    print("[chat] Raw run.content:", repr(run.content))
    if run.status != RunStatus.completed:

        content = str(run.content or "").lower()

        if "prompt injection" in content or "jailbreak" in content:
            failure_reply = BLOCKED_RESPONSE
        else:
            failure_reply = FALLBACK_REPLY

        return {
            "agent_used": agent_name,
            **failure_reply,
        }



    reply = _to_reply(run.content)

    print("[chat] Final response length:", len(reply["response"]))

    return {
        "agent_used": agent_name,
        **reply,
    }
