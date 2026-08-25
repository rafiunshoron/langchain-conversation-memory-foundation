"""
Module 9B — Supabase PostgreSQL Persistent Memory

Purpose:
Replace SQLite memory with PostgreSQL memory.

Concepts:
- PostgreSQL persistence
- Supabase database
- RunnableWithMessageHistory
- SQL-backed memory
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

RAW_DATABASE_URL = os.getenv("SUPABASE_DB_URL")

if not RAW_DATABASE_URL:
    raise ValueError(
        "SUPABASE_DB_URL was not found in the .env file."
    )

DATABASE_URL = RAW_DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)

if DATABASE_URL == RAW_DATABASE_URL:
    raise ValueError(
        "SUPABASE_DB_URL must begin with postgresql://"
    )


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# ---------------------------------------------------------
# PostgreSQL History
# ---------------------------------------------------------

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=DATABASE_URL,
        table_name="message_store",
    )


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Remember previous messages.",
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "{input}",
        ),
    ]
)


# ---------------------------------------------------------
# LCEL Chain
# ---------------------------------------------------------

chain = (
    prompt
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------
# Add PostgreSQL Memory
# ---------------------------------------------------------

chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ---------------------------------------------------------
# Session
# ---------------------------------------------------------

config = {
    "configurable": {
        "session_id": "user_001",
    }
}


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

print("=" * 60)
print("SUPABASE POSTGRES MEMORY")
print("=" * 60)

response = chat.invoke(
    {
        "input": "My name is Alex and I like Python."
    },
    config=config,
)

print("\nAI:")
print(response)

response = chat.invoke(
    {
        "input": "What is my name and what do I like?"
    },
    config=config,
)

print("\nAI:")
print(response)