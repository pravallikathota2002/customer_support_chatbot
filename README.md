# Fleet Management Chatbot

An AI-powered conversational assistant designed to help users with fleet management services, vehicle-related queries, maintenance information, and troubleshooting. The chatbot provides natural, context-aware responses and suggests relevant follow-up questions to make conversations more interactive.

## 🎥 YouTube Demo

Watch the complete chatbot demonstration:

[![Fleet Management Chatbot Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/hqdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

In this demonstration, you can see how the chatbot:

- Greets users in a natural and friendly way.
- Remembers conversational context and responds accordingly.
- Provides relevant follow-up questions based on the conversation.
- Answers questions about available fleet management services.
- Handles user queries through a conversational interface.

## ✨ Key Features

- **Natural Conversations:** Provides friendly, human-like conversational responses.
- **Context Awareness:** Uses previous conversation history to understand follow-up questions.
- **Knowledge-Based Responses:** Retrieves relevant information from the fleet management knowledge base.
- **Multi-Agent Architecture:** Routes user queries to specialized agents based on their intent.
- **Troubleshooting Support:** Helps users investigate vehicle and fleet-related issues.
- **Technician Support:** Supports technician and service-ticket-related workflows.
- **Follow-up Suggestions:** Generates relevant questions to help users continue the conversation.
- **Topic Handling:** Identifies unrelated questions and responds with a predefined Fleet Management Assistant message.

## 🧠 Agent Architecture

The chatbot uses an agent-based architecture to handle different types of user requests.

| Agent | Responsibility |
|---|---|
| Knowledge Agent | Answers fleet-related questions using the knowledge base. |
| General Agent | Handles greetings, acknowledgements, and general conversation. |
| Troubleshooting Agent | Helps investigate vehicle and fleet-related problems. |
| Technician Agent | Handles technician and service-ticket-related requests. |
| Off-topic Handler | Returns a predefined response for unrelated questions. |

A query-routing component uses the current message and conversation history to select the appropriate agent.

## 🛠️ Technology Stack

- **Language:** Python
- **Backend:** FastAPI
- **Agent Framework:** Agno
- **LLM Provider:** OpenRouter
- **Model:** Qwen (configurable)
- **Knowledge Retrieval:** Vector-based retrieval
- **Vector Database:** PostgreSQL with pgvector
- **Embeddings:** FastEmbed
- **Query Routing:** TypeSafe SDK

## 🏗️ High-Level Workflow

1. The user sends a message through the chatbot interface.
2. The backend receives the message along with the session context.
3. The query router analyzes the message and conversation history.
4. The appropriate agent is selected.
5. The Knowledge Agent retrieves relevant information when required.
6. The selected agent generates a response.
7. The chatbot returns the response and relevant follow-up questions to the user.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- OpenRouter API key
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and configure the required values:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3-30b-a3b
DB_URL=your_postgresql_connection_string
```

Use your actual database connection details and API credentials. Do not commit secrets to GitHub.

### 5. Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

You can access the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## 💬 Example Questions

Try asking the chatbot:

- What services do you offer?
- How can I track my vehicles?
- What causes increased fuel consumption?
- How can I maintain vehicle performance?
- How do I report a vehicle issue?

## 🔒 Security

- Store API keys and database credentials in environment variables.
- Avoid exposing confidential information in chatbot responses.
- Validate user input and apply appropriate prompt-injection protection.
- Restrict access to sensitive fleet and customer information.

## 🔮 Future Enhancements

- Real-time voice-based conversations.
- Integration with fleet management systems.
- Advanced vehicle analytics and reporting.
- Improved troubleshooting workflows.
- Enhanced human-agent handoff capabilities.

---

**Note:** This project is designed to demonstrate an AI-powered conversational assistant for fleet management use cases. Actual capabilities depend on the configured agents, knowledge base, integrations, and deployment environment.