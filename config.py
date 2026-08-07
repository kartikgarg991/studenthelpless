import os

from dotenv import load_dotenv


load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
MISTRAL_EMBEDDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
# Keep the default at 768 to match existing Pinecone index dimension (Gemini default).
# If your Pinecone index is 1024-D, set MISTRAL_EMBEDDING_DIMENSIONALITY=1024 instead.
MISTRAL_EMBEDDING_DIMENSIONALITY = int(os.getenv("MISTRAL_EMBEDDING_DIMENSIONALITY", 768))

# Backwards-compat (if present in old envs). Not used by the app anymore.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
REDIS_URL = os.getenv("REDIS_URL")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 7200))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 3))

DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME"),
    'port': int(os.getenv("DB_PORT", 3306))
}
