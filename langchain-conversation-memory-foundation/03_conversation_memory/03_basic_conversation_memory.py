"""
Module 1 — Basic Conversation Memory

Purpose:
Understand how conversation memory works internally.

Concepts:
- Manual message history
- HumanMessage
- AIMessage
- Sending previous messages to LLM

No:
- RunnableWithMessageHistory
- Database
- Persistence
"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)


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
# Create conversation history manually
# ---------------------------------------------------------

conversation_history = [

    SystemMessage(
        content="You are a helpful assistant."
    )

]


# ---------------------------------------------------------
# Function to chat with memory
# ---------------------------------------------------------

def chat(user_input: str):

    # 1. Add user message to history

    conversation_history.append(
        HumanMessage(
            content=user_input
        )
    )


    # 2. Send complete history to LLM

    response = llm.invoke(
        conversation_history
    )


    # 3. Store AI response

    conversation_history.append(
        AIMessage(
            content=response.content
        )
    )


    return response.content



# ---------------------------------------------------------
# Conversation
# ---------------------------------------------------------

print("=" * 60)
print("BASIC CONVERSATION MEMORY")
print("=" * 60)


response1 = chat(
    "Hi, my name is Alex."
)

print("\nAI:", response1)



response2 = chat(
    "What is my name?"
)

print("\nAI:", response2)



# ---------------------------------------------------------
# Inspect stored memory
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("CURRENT MEMORY STATE")
print("=" * 60)


for message in conversation_history:

    print(
        f"{message.type}: {message.content}"
    )