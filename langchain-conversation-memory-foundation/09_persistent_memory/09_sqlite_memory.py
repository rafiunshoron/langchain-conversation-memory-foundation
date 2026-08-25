"""
Module 9A — SQLite Persistent Memory

Purpose:
Understand database-backed conversation memory.

Concepts:
- SQLChatMessageHistory
- SQLite persistence
- RunnableWithMessageHistory
- Session based storage
- Memory survives restart

Modern LangChain:
- LCEL
- ChatGroq
- langchain-community
"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
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
# Database configuration
# ---------------------------------------------------------

db_path = "./chat_history.db"


connection_string = (
    f"sqlite:///{db_path}"
)



# ---------------------------------------------------------
# Retrieve history from database
# ---------------------------------------------------------

def get_session_history(
        session_id: str
) -> BaseChatMessageHistory:


    return SQLChatMessageHistory(

        session_id=session_id,

        connection=connection_string

    )



# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            """
You are a helpful assistant.

Remember user preferences
from previous conversations.
"""

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

chain = (

    prompt

    |

    llm

    |

    StrOutputParser()

)



# ---------------------------------------------------------
# Add persistent memory
# ---------------------------------------------------------

chain_with_history = RunnableWithMessageHistory(

    chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="history",

)



# ---------------------------------------------------------
# Session ID
# ---------------------------------------------------------

config = {

    "configurable":

    {

        "session_id":

        "user_001"

    }

}



# ---------------------------------------------------------
# RUN 1
# Store information
# ---------------------------------------------------------

print("=" * 60)
print("RUN 1 — SAVING MEMORY")
print("=" * 60)



messages = [

    "My name is Alex.",

    "I prefer dark mode interfaces.",

]



for message in messages:


    print("\nUser:")
    print(message)


    response = chain_with_history.invoke(

        {

            "input": message

        },

        config=config

    )


    print("\nAI:")
    print(response)



# ---------------------------------------------------------
# Simulate application restart
# ---------------------------------------------------------

print("\n\n")
print("=" * 60)
print("APPLICATION RESTART")
print("=" * 60)


print(
    """
The Python memory is gone.

A new chain will be created.

The database remains.
"""
)



# ---------------------------------------------------------
# RUN 2
# Load old memory
# ---------------------------------------------------------

print("=" * 60)
print("RUN 2 — RECOVER MEMORY")
print("=" * 60)



new_chain = RunnableWithMessageHistory(

    chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="history",

)



question = (
    "What is my name and what interface theme do I prefer?"
)



response = new_chain.invoke(

    {

        "input": question

    },

    config=config

)



print("\nUser:")
print(question)



print("\nAI:")
print(response)