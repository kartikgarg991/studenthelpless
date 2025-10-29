from flask import Flask, request, jsonify
import mysql.connector
import json
import google.generativeai as genai
from pinecone import Pinecone
from flask_cors import CORS
from dotenv import load_dotenv
import os


load_dotenv()
app = Flask(__name__)
CORS(app)

# ============= CONFIGURATION =============

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME"),
    'port': int(os.getenv("DB_PORT", 3306))
}
# ============= GLOBAL CACHE =============
SUBJECTS_CACHE = []
SCHEMA_CACHE = {}

# ============= CONVERSATION QUEUE =============
conversation_queue = []
MAX_HISTORY = 3  # Last 3 Q&A pairs

def add_to_history(user_msg, assistant_msg):
    """Add conversation to queue, auto-keep last 3"""
    global conversation_queue
    
    conversation_queue.append({
        'user': user_msg,
        'assistant': assistant_msg
    })
    
    # ✅ Simple slice: Always keep last 3
    conversation_queue = conversation_queue[-3:]
    
    print(f"📝 Conversation queue size: {len(conversation_queue)}/{MAX_HISTORY}")

def get_context_for_llm():
    """Format last 3 conversations for LLM context"""
    if not conversation_queue:
        return "No previous conversation context."
    
    context = "RECENT CONVERSATION HISTORY:\n"
    for i, conv in enumerate(conversation_queue, 1):
        # Truncate long responses for context
        assistant_preview = conv['assistant'][:150] + "..." if len(conv['assistant']) > 150 else conv['assistant']
        context += f"\n[Turn {i}]\n"
        context += f"User: {conv['user']}\n"
        context += f"Assistant: {assistant_preview}\n"
    
    return context

# ============= DATABASE CONNECTION =============
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None

# ============= CACHE INITIALIZATION =============
def initialize_cache():
    """Load subjects and schema into memory on startup"""
    global SUBJECTS_CACHE, SCHEMA_CACHE
    
    print("\n🔄 Initializing cache...")
    
    # Load subjects
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("SELECT DISTINCT subject_name FROM subjects ORDER BY subject_name")
            SUBJECTS_CACHE = [row[0] for row in cursor.fetchall()]
            cursor.close()
            connection.close()
            print(f"✅ Cached {len(SUBJECTS_CACHE)} subjects")
    except Exception as e:
        print(f"❌ Error caching subjects: {e}")
        SUBJECTS_CACHE = []
    
    # Load schema
    SCHEMA_CACHE = {
        "tables": {
            "branches": ["id", "branch_code", "branch_name"],
            "subjects": ["id", "branch_id", "subject_code", "subject_name", "semester"],
            "syllabus": ["id", "subject_id", "syllabus_pdf_url", "syllabus_text", "units"],
            "subject_pyqs": ["id", "subject_id", "file_url"]
        },
        "relationships": {
            "subjects.branch_id": "branches.id",
            "syllabus.subject_id": "subjects.id",
            "subject_pyqs.subject_id": "subjects.id"
        }
    }
    print(f"✅ Schema cached")

# ============= INTELLIGENT QUERY CLASSIFIER =============
def classify_query_type(user_query):
    """
    Intelligently determines query type using CACHED subjects + conversation context
    """
    # ✅ Use cached subjects instead of DB call
    available_subjects = SUBJECTS_CACHE
    
    # ✅ Get conversation context
    context = get_context_for_llm()
    
    prompt = f"""You are a query classifier for an academic system.

{context}

DATABASE CONTEXT:
Available subjects: {available_subjects}

CURRENT USER QUERY: "{user_query}"

CLASSIFICATION RULES:
1. SQL: Query asks for direct data retrieval
- Examples: "give me links", "show syllabus", "list subjects", "what is in unit X"

2. PINECONE: Query asks for analysis or patterns
- Examples: "how many times", "important topics", "most asked", "is X important", "frequency"

3. INVALID: Query is generic knowledge or unrelated to available subjects
- Examples: "what is machine learning", "how to study", "explain quantum physics"
- Check: Does query relate to any subject in the available subjects list?

Consider the conversation history when classifying follow-up questions.

Think step by step:
1. Is this a follow-up question? Check conversation history
2. Does this query ask for links, syllabus text, or list of subjects? → SQL
3. Does this query ask for analysis, frequency, or importance? → PINECONE
4. Is this query about general knowledge unrelated to our subjects? → INVALID

Return ONLY one word: SQL, PINECONE, or INVALID"""

    try:
        response = model.generate_content(prompt)
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
        print(f"❌ Classification error: {e}")
        return "SQL"

# ============= INTELLIGENT SUBJECT EXTRACTOR =============
def extract_subject_from_query(user_query):
    """
    Extracts subject using CACHED subjects + conversation context
    """
    # ✅ Use cached subjects instead of DB call
    available_subjects = SUBJECTS_CACHE
    
    # ✅ Get conversation context
    context = get_context_for_llm()
    
    prompt = f"""Extract the subject name from this query.

{context}

AVAILABLE SUBJECTS IN DATABASE:
{json.dumps(available_subjects, indent=2)}

CURRENT USER QUERY: "{user_query}"

RULES:
1. Check conversation history - if previous query mentioned a subject, consider it
2. Check if current query mentions ANY of the available subjects (exact or abbreviated)
3. Common abbreviations: SPM = Software Project Management, NN = Neural Networks
4. If subject name found in the list above, return it EXACTLY as written in the list
5. If NOT found or uncertain, return: NONE

Think:
- Was a subject mentioned in recent conversation?
- Is any subject from the available list mentioned in current query?
- Match full names or common abbreviations

Return ONLY the subject name from the list, or NONE:"""

    try:
        response = model.generate_content(prompt)
        subject = response.text.strip()
        
        # Validate against cached subjects
        if subject == "NONE" or subject not in available_subjects:
            print(f"📚 No valid subject detected in query")
            return None
        
        print(f"📚 Detected subject: {subject}")
        return subject
    except Exception as e:
        print(f"❌ Subject extraction error: {e}")
        return None

# ============= PINECONE SEARCH =============
def search_pinecone(user_query, subject_filter=None, top_k=10):
    """
    Searches Pinecone with optional subject filtering
    """
    print(f"🔍 Searching Pinecone: '{user_query}'")
    if subject_filter:
        print(f"   Filter: subject_name = {subject_filter}")
    
    try:
        # Generate embedding
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_query,
            task_type="retrieval_query"
        )
        query_embedding = embedding_result['embedding']
        
        # Build filter
        filter_dict = {}
        if subject_filter:
            filter_dict = {"subject_name": {"$eq": subject_filter}}
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Filter by score threshold
        filtered_matches = [
            match for match in results['matches'] 
            if match['score'] > 0.55
        ]
        
        print(f"✅ Found {len(filtered_matches)} relevant matches (score > 0.55)")
        
        return filtered_matches
        
    except Exception as e:
        print(f"❌ Pinecone search error: {e}")
        return []

# ============= INTELLIGENT RESPONSE GENERATOR =============
def generate_pinecone_response(user_query, pinecone_matches):
    """
    Generates intelligent response using CACHED subjects + conversation context
    """
    if not pinecone_matches:
        return "I couldn't find relevant information in past exam papers for this query. Try rephrasing or ask about a specific subject."
    
    # ✅ Use cached subjects
    available_subjects = SUBJECTS_CACHE
    
    # ✅ Get conversation context
    context = get_context_for_llm()
    
    # Extract unique years from PYQ matches
    unique_years = set()
    for match in pinecone_matches:
        year = match['metadata'].get('exam_year')
        content_type = match['metadata'].get('content_type')
        if content_type == 'PYQ' and year != 'N/A':
            unique_years.add(year)
    
    unique_years_list = sorted(list(unique_years), reverse=True)
    year_count = len(unique_years_list)
    
    # Prepare match summaries
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
    
    # Generate intelligent prompt
    prompt = f"""You are an intelligent academic assistant analyzing exam data.

{context}

CURRENT USER QUERY: "{user_query}"

DATABASE CONTEXT:
- Available subjects in system: {available_subjects}
- These are the ONLY valid subjects

DATA ANALYSIS:
- PYQ matches found: {len([m for m in pinecone_matches if m['metadata'].get('content_type') == 'PYQ'])}
- Years with data: {unique_years_list}
- Total unique years: {year_count}

MATCHED RESULTS:
{json.dumps(match_summaries, indent=2)}

INSTRUCTIONS:
1. Consider conversation history when answering follow-up questions
2. Understand the query context:
   - If query mentions a subject from the available list, that's the SUBJECT being analyzed
   - Otherwise, determine what entity (topic/concept) is being asked about

3. Answer intelligently:
   - For frequency queries: Count unique years and list them
   - For importance queries: Analyze frequency and recency
   - For pattern queries: Identify trends from the data

4. Priority levels:
   - 3+ years = HIGH priority
   - 1-2 years = MEDIUM priority  
   - 0 years = LOW priority

5. Be conversational and student-friendly (3-5 sentences)
6. Always cite specific years when relevant

Generate your intelligent answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ LLM generation error: {e}")
        return "Error generating response. Please try again."

# ============= SQL QUERY GENERATOR =============
def generate_sql_query(user_query):
    """
    Generates SQL using CACHED schema + conversation context
    ✅ Fresh chat each time (no memory leak)
    """
    # ✅ Use cached schema
    schema = SCHEMA_CACHE
    
    # ✅ Get conversation context
    context = get_context_for_llm()
    
    # ✅ Create fresh chat (no global memory leak)
    fresh_chat = model.start_chat()
    
    system_instruction = f"""You are an intelligent SQL generator.

{context}

DATABASE SCHEMA:
{json.dumps(schema, indent=2)}

IMPORTANT NOTES:
- branch_code has values like: 'CSE', 'ECE', 'ME' (short codes)
- branch_name has full names like: 'Computer Science', 'Electronics and Communication'
- When user mentions 'CSE', 'ECE', etc., use branch_code column for matching
- units column in syllabus contains JSON array, select entire column if needed
- Consider conversation history for follow-up queries

Generate ONLY the SQL query, no explanations or markdown.

CURRENT USER QUERY: {user_query}

SQL:"""

    try:
        response = fresh_chat.send_message(system_instruction)
        sql = response.text.strip().replace('```sql', '').replace('```', '').strip()
        print(f"📝 Generated SQL: {sql}")
        return sql
    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        return None

def format_final_answer(user_query, raw_results):
    """
    Formats SQL results intelligently with conversation context
    """
    # ✅ Get conversation context
    context = get_context_for_llm()
    
    prompt = f"""
{context}

CURRENT USER QUERY: "{user_query}"

Database returned:
{json.dumps(raw_results, indent=2)}

Format this data into a helpful, conversational answer.
- Don't manipulate with data 
- Handle JSON fields (like units) if present
- Format links nicely
- Be concise and student-friendly
- Consider conversation history

Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Formatting error: {e}")
        return json.dumps(raw_results)

# ============= MAIN QUERY ENDPOINT =============
@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'success': False, 'error': 'Query required'}), 400
        
        print(f"\n{'='*60}")
        print(f"📩 Received query: {user_query}")
        print(f"📝 Current queue size: {len(conversation_queue)}/{MAX_HISTORY}")
        print(f"{'='*60}")
        
        # ===== INTELLIGENT CLASSIFICATION =====
        query_type = classify_query_type(user_query)
        
        # ===== HANDLE INVALID QUERIES =====
        if query_type == "INVALID":
            print("⚠️ Query classified as INVALID\n")
            invalid_response = 'This query appears to be general knowledge or unrelated to available subjects.'
            
            # ✅ Add to history even for invalid queries
            add_to_history(user_query, invalid_response)
            
            return jsonify({
                'success': False,
                'type': 'INVALID',
                'error': invalid_response,
                'suggestion': 'Try asking about: past papers, syllabus, topic frequency, or exam patterns for specific subjects.'
            }), 400
        
        # ===== SQL QUERIES =====
        if query_type == "SQL":
            print("🔄 Processing as SQL query...")
            
            sql_query = generate_sql_query(user_query)
            if not sql_query:
                return jsonify({'success': False, 'error': 'Could not generate SQL query'}), 500
            
            connection = get_db_connection()
            if not connection:
                return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(sql_query)
                raw_results = cursor.fetchall()
                cursor.close()
                connection.close()
                
                final_answer = format_final_answer(user_query, raw_results)
                
                # ✅ Add to conversation history
                add_to_history(user_query, final_answer)
                
                print("✅ SQL query completed successfully\n")
                
                return jsonify({
                    'success': True,
                    'type': 'SQL',
                    'query': user_query,
                    'answer': final_answer,
                    'raw_data': raw_results,
                    'context_size': len(conversation_queue)
                })
            except mysql.connector.Error as e:
                print(f"❌ SQL execution error: {e}\n")
                return jsonify({
                    'success': False,
                    'error': 'Could not execute query. Please rephrase your question.'
                }), 500
        
        # ===== PINECONE QUERIES =====
        elif query_type == "PINECONE":
            print("🔄 Processing as PINECONE query...")
            
            # Intelligent subject extraction (with context)
            subject_filter = extract_subject_from_query(user_query)
            
            # Search Pinecone
            pinecone_matches = search_pinecone(user_query, subject_filter, top_k=10)
            
            # Generate intelligent response (with context)
            final_answer = generate_pinecone_response(user_query, pinecone_matches)
            
            # ✅ Add to conversation history
            add_to_history(user_query, final_answer)
            
            print("✅ Pinecone query completed successfully\n")
            
            return jsonify({
                'success': True,
                'type': 'PINECONE',
                'query': user_query,
                'answer': final_answer,
                'matches_found': len(pinecone_matches),
                'subject_filter': subject_filter,
                'context_size': len(conversation_queue),
                'top_matches': [
                    {
                        'subject': m['metadata'].get('subject_name'),
                        'year': m['metadata'].get('exam_year'),
                        'score': round(m['score'], 3),
                        'type': m['metadata'].get('content_type')
                    }
                    for m in pinecone_matches[:5]
                ]
            })
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

# ============= HEALTH CHECK =============
@app.route('/health', methods=['GET'])
def health():
    # Test database connection
    db_status = 'active' if get_db_connection() else 'inactive'
    
    return jsonify({
        'status': 'healthy',
        'services': {
            'database': db_status,
            'pinecone': 'active',
            'gemini': 'active'
        },
        'cache_stats': {
            'subjects_cached': len(SUBJECTS_CACHE),
            'subjects_sample': SUBJECTS_CACHE[:5] if SUBJECTS_CACHE else [],
            'schema_cached': bool(SCHEMA_CACHE)
        },
        'conversation_stats': {
            'queue_size': len(conversation_queue),
            'max_history': MAX_HISTORY
        }
    })

# ============= MAIN =============
if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("🚀 STUDENTHELPER BACKEND - OPTIMIZED VERSION")
    print("="*60)
    print("\n📌 Optimizations:")
    print("   ✅ Cached subjects (no repeated DB calls)")
    print("   ✅ Cached schema (faster SQL generation)")
    print("   ✅ Rolling conversation queue (last 3 exchanges)")
    print("   ✅ Context-aware responses")
    print("   ✅ No memory leaks (fresh SQL chats)")
    print("\n📌 Endpoints:")
    print("   POST /query         - Intelligent query handler")
    print("   GET  /health        - System health check")
    print("   POST /clear-history - Clear conversation queue")
    
    # ✅ Initialize cache on startup
    initialize_cache()
    
    if SUBJECTS_CACHE:
        print(f"\n📚 Sample subjects: {SUBJECTS_CACHE[:3]}")
    
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
