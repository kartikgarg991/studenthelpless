from pinecone import Pinecone

from config import (
    MISTRAL_EMBEDDING_DIMENSIONALITY,
    MISTRAL_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from services.mistral_client import embed_text
from services.usage_tracker import record_call


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def search_pinecone(user_query, subject_filter=None, top_k=10):
    print(f"🔍 Searching Pinecone: '{user_query}'")
    try:
        query_embedding = embed_text(user_query)
        record_call('mistral', success=True)
    except Exception as e:
        record_call('mistral', success=False, error=e)
        print(f"❌ Mistral embedding error: {e}")
        return []

    if len(query_embedding) != MISTRAL_EMBEDDING_DIMENSIONALITY:
        print(
            "⚠️ Embedding dimensionality mismatch. "
            f"Got {len(query_embedding)} but configured {MISTRAL_EMBEDDING_DIMENSIONALITY}. "
            "Update MISTRAL_EMBEDDING_DIMENSIONALITY to match your Pinecone index dimension."
        )
    filter_dict = {"subject_name": {"$eq": subject_filter}} if subject_filter else None

    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        record_call('pinecone', success=True)
        filtered_matches = [m for m in results['matches'] if m['score'] > 0.55]
        print(f"✅ Found {len(filtered_matches)} relevant matches")
        return filtered_matches
    except Exception as e:
        record_call('pinecone', success=False, error=e)
        msg = str(e).lower()
        if "dimension" in msg and ("mismatch" in msg or "invalid" in msg):
            print(
                "❌ Pinecone dimension mismatch. Your Pinecone index dimension must match "
                "the embedding dimension. Either recreate/re-embed the index for Mistral (usually 1024), "
                "or set MISTRAL_EMBEDDING_DIMENSIONALITY to your existing index dimension."
            )
        else:
            print(f"❌ Pinecone search error: {e}")
        return []
