"""
Module 8 — Summary Memory

Purpose:
Understand summary-based conversation memory.

Concepts:
- Running summary
- Recent message buffer
- Compressing old messages
- Summary + recent context architecture

Modern LangChain:
- LCEL
- ChatGroq
- Message based architecture

"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser



# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()



# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

chat_llm = ChatGroq(

    model="openai/gpt-oss-20b",

    temperature=0

)


summary_llm = ChatGroq(

    model="openai/gpt-oss-20b",

    temperature=0

)



# ---------------------------------------------------------
# Conversation Prompt
# ---------------------------------------------------------

chat_prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",
            """
You are a helpful assistant.

Previous conversation summary:

{summary}
"""
        ),


        MessagesPlaceholder(
            variable_name="recent_messages"
        ),


        (
            "human",
            "{input}"
        ),

    ]

)



# ---------------------------------------------------------
# Summarization Prompt
# ---------------------------------------------------------

summary_prompt = ChatPromptTemplate.from_template(

"""
Create a short summary of the conversation.

Preserve important user facts,
preferences, and context.

Current summary:

{current_summary}


New messages:

{new_messages}


Updated summary:
"""

)



# ---------------------------------------------------------
# Create chains
# ---------------------------------------------------------

chat_chain = (
    chat_prompt
    | chat_llm
    | StrOutputParser()
)



summary_chain = (
    summary_prompt
    | summary_llm
    | StrOutputParser()
)



# ---------------------------------------------------------
# Memory State
# ---------------------------------------------------------

running_summary = ""

recent_messages = []


MAX_RECENT_MESSAGES = 4



# ---------------------------------------------------------
# Conversation Function
# ---------------------------------------------------------

def chat(user_input):

    global running_summary
    global recent_messages


    # Generate response

    response = chat_chain.invoke(

        {

            "summary":
                running_summary
                if running_summary
                else "No previous information.",


            "recent_messages":
                recent_messages,


            "input":
                user_input,

        }

    )


    # Store new messages

    recent_messages.append(

        HumanMessage(
            content=user_input
        )

    )


    recent_messages.append(

        AIMessage(
            content=response
        )

    )


    # Check if summarization is needed

    if len(recent_messages) > MAX_RECENT_MESSAGES:


        old_messages = recent_messages[:-MAX_RECENT_MESSAGES]


        formatted_messages = "\n".join(

            [

                f"{message.type}: {message.content}"

                for message in old_messages

            ]

        )


        # Update summary

        running_summary = summary_chain.invoke(

            {

                "current_summary":
                    running_summary
                    if running_summary
                    else "None",


                "new_messages":
                    formatted_messages,

            }

        )


        # Keep recent messages only

        recent_messages = recent_messages[-MAX_RECENT_MESSAGES:]


    return response



# ---------------------------------------------------------
# Test Conversation
# ---------------------------------------------------------

print("=" * 60)
print("SUMMARY MEMORY")
print("=" * 60)



conversation = [

    "My name is Alex and I live in Dhaka.",

    "I work as an AI engineer.",

    "I am learning LangChain and RAG.",

    "I prefer concise explanations.",

    "What do you know about me?"

]



for message in conversation:


    print("\nUser:")
    print(message)


    response = chat(message)


    print("\nAI:")
    print(response)



# ---------------------------------------------------------
# Final Memory State
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL MEMORY STATE")
print("=" * 60)


print("\nSummary:")

print(running_summary)



print("\nRecent Messages:")


for message in recent_messages:

    print(

        message.type,
        ":",
        message.content

    )