"""Uses the typesafe model to pick between the general and knowledge agents."""

import os

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

from chatbot.services.context_history import get_previous

load_dotenv()

client = TypeSafeClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api",
    model="~typesafe/jev-latest",
)

DEFAULT_AGENT = "knowledge_agent"
MIN_CONFIDENCE = 0.55   # below this, a vague message stays with the previous agent

# need to add jailbreak agent

AGENT_CRITERIA = {
    "knowledge_agent": (
        "Questions related to fleet management, vehicles, fuel management, "
        "mileage, maintenance, EV charging, car wash, services, products, "
        "pricing, policies, manuals, procedures, driver monitoring and "
        "device installation. Also contextual follow-ups related to these topics. "
        "For example, 'give me instructions' after discussing a fleet-related process. "
        "The first message of a new fleet-related problem must go here."
    ),

    "general_agent": (
        "Only greetings, thanks, goodbyes, acknowledgements, small talk, "
        "and questions like 'what can you help me with?'"
    ),

    "off_topic_agent": (
        "Questions or requests unrelated to fleet management and the services "
        "supported by this assistant. This includes general knowledge, geography, "
        "cooking, recipes, entertainment, unrelated programming, and other "
        "unrelated topics. Examples: 'What is the capital of India?', "
        "'How do I make coffee?', 'Tell me a joke'. "
        "Use the current message and previous conversation to determine relevance. "
        "A new unrelated question remains off-topic even if the previous "
        "conversation was about fleet management."
    ),

    "troubleshooting_agent": (
        "Continuing a diagnosis after the knowledge base has already answered. "
        "The user answers diagnostic questions, says the suggestions did not help, "
        "or agrees to troubleshoot."
    ),

    "technician_agent": (
        "The user wants to raise, view, update, resolve, reopen or cancel "
        "a fault ticket or work order, check ticket status, or request a technician."
    )
}

ROUTING_INSTRUCTIONS = (
    "Choose the correct agent based on the current message and conversation history. "

    "Use previous_user_message, previous_answer, previous_follow_up_questions "
    "and previous_agent to understand contextual follow-ups. "

    "Rules in priority order: "

    "1. If the current message is unrelated to fleet management, "
    "vehicle management or supported services, choose off_topic_agent. "

    "2. If the user refers to a previous fleet-related question or follow-up, "
    "continue the relevant fleet-related conversation. "

    "3. Ticket creation, ticket status, ticket updates, cancellation, "
    "or technician requests -> technician_agent. "

    "4. If the user answers a diagnostic question or agrees to continue "
    "troubleshooting -> troubleshooting_agent. "

    "5. New fleet-related questions, service questions, how-to questions, "
    "cost questions and new fleet-related problems -> knowledge_agent. "

    "6. Pure greetings, thanks, goodbyes and small talk -> general_agent. "

    "7. If the user says 'give me instructions', 'how do I do that?', "
    "'what about the cost?' or another vague follow-up, use chat history "
    "to identify the subject. "

    "8. If the message is a new unrelated question, do not use the previous "
    "fleet-related conversation to force it into knowledge_agent. "

    "If unsure whether a message is related to fleet management, "
    "choose off_topic_agent rather than inventing a fleet-related answer."
)

def get_agent_for_query(query: str, session_id: str) -> dict:
    """Return {'agent': 'knowledge_agent' | 'general_agent', 'probabilities': ...}."""
    previous = get_previous(session_id)

    response = client.system_one(
            state={
            "message": query,
            "previous_user_message": previous["question"],
            "previous_answer": previous["answer"],
            "previous_agent": previous["agent"],
            "previous_follow_up_questions": previous["follow_ups"],
        },
        questions={
            "agent": Choice(
                instructions=ROUTING_INSTRUCTIONS,
                criteria=AGENT_CRITERIA,
            )
        },
    )

    answer = response.answers["agent"]
    agent = answer.choice
    probabilities = answer.probabilities

    # Low confidence: keep the previous agent if there is one
    best_score = max(probabilities.values()) if isinstance(probabilities, dict) and probabilities else 1.0
    if best_score < MIN_CONFIDENCE and previous["agent"] in AGENT_CRITERIA:
        agent = previous["agent"]

    return {"agent": agent, "probabilities": probabilities}

