from __future__ import annotations

from typing import Optional

from mistralai import Mistral

from config import (
    MISTRAL_API_KEY,
    MISTRAL_CHAT_MODEL,
    MISTRAL_EMBEDDING_DIMENSIONALITY,
    MISTRAL_EMBEDDING_MODEL,
)

def _get_client() -> Mistral:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is missing")
    return Mistral(api_key=MISTRAL_API_KEY)


def chat_complete(prompt: str, *, system: Optional[str] = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = _get_client()
    res = client.chat.complete(model=MISTRAL_CHAT_MODEL, messages=messages)
    return (res.choices[0].message.content or "").strip()


def embed_text(text: str) -> list[float]:
    client = _get_client()

    # Try to keep Pinecone compatibility when a custom dimension is desired.
    try:
        res = client.embeddings.create(
            model=MISTRAL_EMBEDDING_MODEL,
            inputs=[text],
            output_dimension=MISTRAL_EMBEDDING_DIMENSIONALITY,
        )
    except Exception:
        res = client.embeddings.create(
            model=MISTRAL_EMBEDDING_MODEL,
            inputs=[text],
        )

    embedding = res.data[0].embedding
    return list(embedding)

