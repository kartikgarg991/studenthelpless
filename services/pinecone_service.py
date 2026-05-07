import google.generativeai as genai
from pinecone import Pinecone

from config import (
    GEMINI_EMBEDDING_DIMENSIONALITY,
    GEMINI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from services.usage_tracker import record_call


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def search_pinecone(user_query, subject_filter=None, top_k=10):
    print(f"🔍 Searching Pinecone: '{user_query}'")
    try:
        embedding_result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=user_query,
            task_type="retrieval_query",
            output_dimensionality=GEMINI_EMBEDDING_DIMENSIONALITY
        )
        record_call('gemini', success=True)
    except Exception as e:
        record_call('gemini', success=False, error=e)
        print(f"❌ Gemini embedding error: {e}")
        return []

    query_embedding = embedding_result['embedding']
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
        print(f"❌ Pinecone search error: {e}")
        return []
