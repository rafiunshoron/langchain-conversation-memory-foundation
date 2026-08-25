"""
Module 9B.1 — Supabase PostgreSQL Connection Test

Purpose:
Test Python connection with Supabase PostgreSQL.

Concepts:
- Environment variables
- PostgreSQL connection
- psycopg driver

No LangChain yet.
"""


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import os

from dotenv import load_dotenv

import psycopg



# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()



# ---------------------------------------------------------
# Get database URL
# ---------------------------------------------------------

database_url = os.getenv(
    "SUPABASE_DB_URL"
)



# ---------------------------------------------------------
# Connect to PostgreSQL
# ---------------------------------------------------------

connection = psycopg.connect(
    database_url
)



# ---------------------------------------------------------
# Test query
# ---------------------------------------------------------

cursor = connection.cursor()


cursor.execute(
    "SELECT version();"
)


database_version = cursor.fetchone()



print("=" * 60)
print("CONNECTED SUCCESSFULLY")
print("=" * 60)


print(database_version[0])



# ---------------------------------------------------------
# Close connection
# ---------------------------------------------------------

cursor.close()

connection.close()


print("\nConnection closed.")