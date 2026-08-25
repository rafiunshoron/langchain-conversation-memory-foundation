"""
Module 7 — Window Memory

Purpose:
Understand fixed-size conversation memory.

Concepts:
- Custom ChatMessageHistory
- Sliding window memory
- Keeping last K exchanges
- Removing old messages automatically

Modern LangChain + Pydantic compatible version
"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)



# ---------------------------------------------------------
# Windowed Memory Implementation
# ---------------------------------------------------------

class WindowedChatHistory(InMemoryChatMessageHistory):
    """
    Keeps only the latest K conversation exchanges.

    One exchange:
        HumanMessage
        +
        AIMessage
    """


    k: int = 2


    def add_messages(self, messages):

        # Add incoming messages

        super().add_messages(messages)


        # Each exchange contains:
        # Human + AI = 2 messages

        max_messages = self.k * 2


        # Keep only the latest window

        if len(self.messages) > max_messages:

            self.messages = self.messages[-max_messages:]



# ---------------------------------------------------------
# Create memory window
# ---------------------------------------------------------

memory = WindowedChatHistory(
    k=2
)



# ---------------------------------------------------------
# Simulated conversation
# ---------------------------------------------------------

conversation = [

    (
        "My name is Alex.",
        "Nice to meet you Alex."
    ),

    (
        "I live in Dhaka.",
        "Dhaka is a busy city."
    ),

    (
        "I work as an AI engineer.",
        "That sounds interesting."
    ),

    (
        "I have two cats.",
        "Cats make great companions."
    ),

]



# ---------------------------------------------------------
# Add messages
# ---------------------------------------------------------

print("=" * 60)
print("WINDOW MEMORY SIMULATION")
print("=" * 60)


for user_message, ai_message in conversation:


    memory.add_messages(
        [

            HumanMessage(
                content=user_message
            ),

            AIMessage(
                content=ai_message
            ),

        ]
    )


    print("\nCurrent Memory:")

    for message in memory.messages:

        print(
            f"{message.type}: {message.content}"
        )


    print("-" * 60)



# ---------------------------------------------------------
# Final memory
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL MEMORY STATE")
print("=" * 60)


for message in memory.messages:

    print(
        f"{message.type}: {message.content}"
    )