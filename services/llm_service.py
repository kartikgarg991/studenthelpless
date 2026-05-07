import json

from services.cache_service import get_schema_cache, get_subjects_cache
from services.gemini_client import model
from services.session_store import get_context_for_llm
from services.usage_tracker import record_call


def classify_query_type(user_query, session_id):
    available_subjects = get_subjects_cache()
    context = get_context_for_llm(session_id)
    prompt = f"""You are a query classifier for an academic system.

{context}

DATABASE CONTEXT:
Available subjects: {available_subjects}

CURRENT USER QUERY: "{user_query}"

CLASSIFICATION RULES:
1. SQL: Query asks for direct data retrieval
2. PINECONE: Query asks for analysis or patterns
3. INVALID: Query is generic knowledge or unrelated to available subjects

Return ONLY one word: SQL, PINECONE, or INVALID"""

    try:
        response = model.generate_content(prompt)
        record_call('gemini', success=True)
        result = response.text.strip().upper()
        if "INVALID" in result:
            classification = "INVALID"
        elif "SQL" in result:
            classification = "SQL"
        else:
            classification = "PINECONE"
        print(f"🔍 Query classified as: {classification}")
        return classification
    except Exception as e:
        record_call('gemini', success=False, error=e)
        print(f"❌ Classification error: {e}")
        return "SQL"


def extract_subject_from_query(user_query, session_id):
    available_subjects = get_subjects_cache()
    context = get_context_for_llm(session_id)
    prompt = f"""Extract the subject name from this query.

{context}

AVAILABLE SUBJECTS IN DATABASE:
{json.dumps(available_subjects, indent=2)}

CURRENT USER QUERY: "{user_query}"

Return ONLY the subject name from the list, or NONE:"""

    try:
        response = model.generate_content(prompt)
        record_call('gemini', success=True)
        subject = response.text.strip()
        if subject == "NONE" or subject not in available_subjects:
            print("📚 No valid subject detected in query")
            return None
        print(f"📚 Detected subject: {subject}")
        return subject
    except Exception as e:
        record_call('gemini', success=False, error=e)
        print(f"❌ Subject extraction error: {e}")
        return None


def generate_pinecone_response(user_query, pinecone_matches, session_id):
    if not pinecone_matches:
        return "I couldn't find relevant information in past exam papers for this query."
    available_subjects = get_subjects_cache()
    context = get_context_for_llm(session_id)
    unique_years = set()
    for match in pinecone_matches:
        year = match['metadata'].get('exam_year')
        if match['metadata'].get('content_type') == 'PYQ' and year != 'N/A':
            unique_years.add(year)
    unique_years_list = sorted(list(unique_years), reverse=True)
    match_summaries = []
    for i, match in enumerate(pinecone_matches[:10], 1):
        metadata = match['metadata']
        match_summaries.append({
            'rank': i,
            'score': round(match['score'], 3),
            'subject': metadata.get('subject_name', 'Unknown'),
            'year': metadata.get('exam_year', 'N/A'),
            'type': metadata.get('content_type', 'N/A'),
            'preview': metadata.get('text', '')[:200]
        })
    prompt = f"""You are an intelligent academic assistant analyzing exam data.

{context}

CURRENT USER QUERY: "{user_query}"
Available subjects: {available_subjects}
Years with data: {unique_years_list}
MATCHED RESULTS: {json.dumps(match_summaries, indent=2)}

Be conversational and student-friendly (3-5 sentences). Always cite specific years.
Generate your intelligent answer:"""

    try:
        response = model.generate_content(prompt)
        record_call('gemini', success=True)
        return response.text.strip()
    except Exception as e:
        record_call('gemini', success=False, error=e)
        print(f"❌ LLM generation error: {e}")
        return "Error generating response. Please try again."


def generate_sql_query(user_query, session_id):
    schema = get_schema_cache()
    context = get_context_for_llm(session_id)
    fresh_chat = model.start_chat()
    system_instruction = f"""You are an intelligent SQL generator.

{context}

DATABASE SCHEMA: {json.dumps(schema, indent=2)}

Generate ONLY the SQL query, no explanations or markdown.
CURRENT USER QUERY: {user_query}
SQL:"""

    try:
        response = fresh_chat.send_message(system_instruction)
        record_call('gemini', success=True)
        sql = response.text.strip().replace('```sql', '').replace('```', '').strip()
        print(f"📝 Generated SQL: {sql}")
        return sql
    except Exception as e:
        record_call('gemini', success=False, error=e)
        print(f"❌ SQL generation error: {e}")
        return None


def format_final_answer(user_query, raw_results, session_id):
    context = get_context_for_llm(session_id)
    prompt = f"""
{context}

CURRENT USER QUERY: "{user_query}"
Database returned: {json.dumps(raw_results, indent=2)}

Format this into a helpful, student-friendly answer using ONLY the database returned data.

STRICT RULES:
- Do not use general knowledge.
- Do not invent syllabus, units, links, years, subjects, or explanations.
- Do not output raw JSON, Python lists, escaped newline text, or database-looking blobs.
- If a URL field exists, format it as a clean Markdown link.
- If syllabus_text or units exist, format them into readable sections and bullet points.
- If the data contains units like "UNIT 1", "UNIT 2", etc., preserve those unit names as headings.
- If the database returned a file_url or syllabus_pdf_url, make it easy to click.
- Keep the answer concise, clean, and student-friendly.
- If a field is missing, simply omit it.

Answer:"""
    try:
        response = model.generate_content(prompt)
        record_call('gemini', success=True)
        return response.text.strip()
    except Exception as e:
        record_call('gemini', success=False, error=e)
        print(f"❌ Formatting error: {e}")
        return json.dumps(raw_results)
