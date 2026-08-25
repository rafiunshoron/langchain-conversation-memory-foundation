"""
Module 4 — RunnableWithMessageHistory

Purpose:
Learn LangChain's modern conversation memory approach.

Concepts:
- RunnableWithMessageHistory
- InMemoryChatMessageHistory
- Session based memory
- MessagesPlaceholder
- LCEL + Memory

"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from typing import Dict

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.output_parsers import StrOutputParser



# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()



# ---------------------------------------------------------
# Create LLM
# ---------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)



# ---------------------------------------------------------
# Create Prompt Template
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            "You are a helpful assistant. Remember user information."
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{input}"
        ),

    ]
)



# ---------------------------------------------------------
# Create LCEL Chain
# ---------------------------------------------------------

chain = prompt | llm | StrOutputParser()



# ---------------------------------------------------------
# Memory Store
# ---------------------------------------------------------

store: Dict[str, BaseChatMessageHistory] = {}



# ---------------------------------------------------------
# Function to retrieve user history
# ---------------------------------------------------------

def get_session_history(session_id: str) -> BaseChatMessageHistory:

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]



# ---------------------------------------------------------
# Add memory capability to chain
# ---------------------------------------------------------

chain_with_history = RunnableWithMessageHistory(

    chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="history",

)



# ---------------------------------------------------------
# User session configuration
# ---------------------------------------------------------

config = {

    "configurable": {

        "session_id": "user_123"

    }

}



# ---------------------------------------------------------
# Conversation
# ---------------------------------------------------------

print("=" * 60)
print("RUNNABLE WITH MESSAGE HISTORY")
print("=" * 60)



messages = [

    "My name is Alex.",

    "I am learning LangChain.",

    "What is my name and what am I learning?"

]



for message in messages:

    print("\nUser:", message)


    response = chain_with_history.invoke(

        {
            "input": message
        },

        config=config

    )


    print("AI:", response)



# ---------------------------------------------------------
# Inspect stored history
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("STORED MESSAGE HISTORY")
print("=" * 60)


history = store["user_123"]


for message in history.messages:

    print(
        f"{message.type}: {message.content}"
    )