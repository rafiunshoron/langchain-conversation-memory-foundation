"""
Module 0.2 — LangChain Expression Language (LCEL)

Purpose:
Understand the modern LangChain pipeline structure.

Concepts covered:
- ChatPromptTemplate
- ChatGroq
- StrOutputParser
- LCEL pipe operator
- Chain invocation

Flow:

Input
  |
Prompt
  |
LLM
  |
Output Parser
  |
String Output
"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# 1. Create the Chat Model
# ---------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ---------------------------------------------------------
# 2. Create Prompt Template
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer clearly."
        ),
        (
            "human",
            "{question}"
        ),
    ]
)


# ---------------------------------------------------------
# 3. Create Output Parser
# ---------------------------------------------------------

parser = StrOutputParser()


# ---------------------------------------------------------
# 4. Build LCEL Pipeline
# ---------------------------------------------------------

chain = prompt | llm | parser


# ---------------------------------------------------------
# 5. Invoke the Chain
# ---------------------------------------------------------

response = chain.invoke(
    {
        "question": "Explain what LangChain is in one paragraph."
    }
)


# ---------------------------------------------------------
# 6. Display Result
# ---------------------------------------------------------

print("=" * 60)
print("LCEL RESPONSE")
print("=" * 60)

print(response)