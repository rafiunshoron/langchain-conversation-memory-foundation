"""
Module 0.1 — LangChain Message System

Purpose:
Understand how LangChain represents conversations internally.

Concepts covered:
- SystemMessage
- HumanMessage
- AIMessage
- Message roles
- Conversation history structure

No LLM.
No memory.
No chains.

This module only focuses on the message layer.
"""


# ---------------------------------------------------------
# Import LangChain message classes
# ---------------------------------------------------------

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


# ---------------------------------------------------------
# 1. Creating individual messages
# ---------------------------------------------------------

system_message = SystemMessage(
    content="You are a helpful AI assistant."
)


human_message = HumanMessage(
    content="What is LangChain?"
)


ai_message = AIMessage(
    content="LangChain is a framework for building applications with Large Language Models."
)


# ---------------------------------------------------------
# 2. Inspecting message objects
# ---------------------------------------------------------

print("=" * 60)
print("INDIVIDUAL MESSAGE OBJECTS")
print("=" * 60)


messages = [
    system_message,
    human_message,
    ai_message,
]


for message in messages:
    print("\nMessage Type:")
    print(message.type)

    print("Content:")
    print(message.content)



# ---------------------------------------------------------
# 3. Building a conversation history
# ---------------------------------------------------------

conversation_history = [

    SystemMessage(
        content="You are a Python programming teacher."
    ),

    HumanMessage(
        content="What is a Python list?"
    ),

    AIMessage(
        content="A Python list is an ordered collection of items."
    ),

    HumanMessage(
        content="Can you give an example?"
    ),

    AIMessage(
        content="Example: numbers = [1, 2, 3, 4]"
    ),
]


# ---------------------------------------------------------
# 4. Reading conversation history
# ---------------------------------------------------------

print("\n\n")
print("=" * 60)
print("CONVERSATION HISTORY")
print("=" * 60)


for index, message in enumerate(conversation_history, start=1):

    print(f"\nMessage {index}")
    print("----------------")

    print("Role:", message.type)
    print("Content:", message.content)



# ---------------------------------------------------------
# 5. Understanding the structure
# ---------------------------------------------------------

print("\n\n")
print("=" * 60)
print("MESSAGE FLOW")
print("=" * 60)

print(
    """
SystemMessage
      |
      ↓
HumanMessage
      |
      ↓
AIMessage
      |
      ↓
HumanMessage
      |
      ↓
AIMessage


This list of messages is the foundation
of LangChain conversation memory.
"""
)