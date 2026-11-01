"""One shared PostgreSQL connection for sessions and chat history."""

import os

from agno.db.postgres import PostgresDb
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL is missing in .env")

db = PostgresDb(
    db_url=DB_URL,
    session_table="agent_sessions",
)