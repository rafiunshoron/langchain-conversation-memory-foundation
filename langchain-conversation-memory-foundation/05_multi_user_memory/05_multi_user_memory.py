"""
Module 5 — Multi User Memory

Purpose:
Understand how LangChain separates conversation
memory between multiple users.

Concepts:
- session_id
- Multiple message histories
- Memory isolation
- RunnableWithMessageHistory

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
# Prompt Template
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            "You are a helpful assistant. Remember user preferences."
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
# LCEL Chain
# ---------------------------------------------------------

chain = prompt | llm | StrOutputParser()



# ---------------------------------------------------------
# Memory Store
#
# Each session_id gets separate history
# ---------------------------------------------------------

store: Dict[str, BaseChatMessageHistory] = {}



# ---------------------------------------------------------
# Retrieve memory for a user
# ---------------------------------------------------------

def get_session_history(session_id: str):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()


    return store[session_id]



# ---------------------------------------------------------
# Add memory capability
# ---------------------------------------------------------

chain_with_history = RunnableWithMessageHistory(

    chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="history",

)



# ---------------------------------------------------------
# Create user sessions
# ---------------------------------------------------------

user_a_config = {

    "configurable": {

        "session_id": "user_a"

    }

}


user_b_config = {

    "configurable": {

        "session_id": "user_b"

    }

}



# ---------------------------------------------------------
# User A conversation
# ---------------------------------------------------------

print("=" * 60)
print("USER A")
print("=" * 60)


response = chain_with_history.invoke(

    {
        "input":
        "My favorite programming language is Python."
    },

    config=user_a_config

)


print("AI:", response)



# ---------------------------------------------------------
# User B conversation
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("USER B")
print("=" * 60)


response = chain_with_history.invoke(

    {
        "input":
        "My favorite programming language is JavaScript."
    },

    config=user_b_config

)


print("AI:", response)



# ---------------------------------------------------------
# Testing memory separation
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("MEMORY TEST")
print("=" * 60)



response_a = chain_with_history.invoke(

    {
        "input":
        "What is my favorite programming language?"
    },

    config=user_a_config

)


print("\nUser A asks:")
print(response_a)



response_b = chain_with_history.invoke(

    {
        "input":
        "What is my favorite programming language?"
    },

    config=user_b_config

)


print("\nUser B asks:")
print(response_b)



# ---------------------------------------------------------
# Inspect stored memories
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("MEMORY STORAGE")
print("=" * 60)



for user_id, history in store.items():

    print("\nSession:", user_id)

    for message in history.messages:

        print(
            message.type,
            ":",
            message.content
        )