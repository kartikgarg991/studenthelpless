from flask import Blueprint, jsonify, request
import mysql.connector

from services.db_fetch import execute_sql_query, get_db_connection
from services.llm_service import (
    classify_query_type,
    extract_subject_from_query,
    format_final_answer,
    generate_pinecone_response,
    generate_sql_query,
)
from services.pinecone_service import search_pinecone
from services.session_store import (
    MAX_HISTORY,
    add_to_history,
    cleanup_sessions,
    get_context_size,
)


query_bp = Blueprint('query', __name__)


@query_bp.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_query:
            return jsonify({'success': False, 'error': 'Query required'}), 400

        cleanup_sessions()

        print(f"\n{'='*60}")
        print(f"📩 Query: {user_query}")
        print(f"👤 Session: {session_id[:8]}...")
        print(f"📝 Queue size: {get_context_size(session_id)}/{MAX_HISTORY}")
        print(f"{'='*60}")

        query_type = classify_query_type(user_query, session_id)

        if query_type == "INVALID":
            invalid_response = 'This query appears to be general knowledge or unrelated to available subjects.'
            add_to_history(session_id, user_query, invalid_response)
            return jsonify({
                'success': False,
                'type': 'INVALID',
                'error': invalid_response,
                'suggestion': 'Try asking about: past papers, syllabus, topic frequency, or exam patterns.'
            }), 400

        if query_type == "SQL":
            print("🔄 Processing as SQL query...")
            sql_query = generate_sql_query(user_query, session_id)
            if not sql_query:
                return jsonify({'success': False, 'error': 'Could not generate SQL query'}), 500
            connection = get_db_connection()
            if not connection:
                return jsonify({'success': False, 'error': 'Database connection failed'}), 500
            try:
                raw_results = execute_sql_query(connection, sql_query)
                if not raw_results:
                    final_answer = "I couldn't find matching data in the database for your query. Please check the subject name or try asking in a slightly different way."
                else:
                    final_answer = format_final_answer(user_query, raw_results, session_id)
                add_to_history(session_id, user_query, final_answer)
                print("✅ SQL query completed\n")
                return jsonify({
                    'success': True,
                    'type': 'SQL',
                    'query': user_query,
                    'answer': final_answer,
                    'raw_data': raw_results,
                    'context_size': get_context_size(session_id)
                })
            except mysql.connector.Error as e:
                print(f"❌ SQL execution error: {e}\n")
                return jsonify({'success': False, 'error': 'Could not execute query. Please rephrase.'}), 500

        elif query_type == "PINECONE":
            print("🔄 Processing as PINECONE query...")
            subject_filter = extract_subject_from_query(user_query, session_id)
            pinecone_matches = search_pinecone(user_query, subject_filter, top_k=10)
            final_answer = generate_pinecone_response(user_query, pinecone_matches, session_id)
            add_to_history(session_id, user_query, final_answer)
            print("✅ Pinecone query completed\n")
            return jsonify({
                'success': True,
                'type': 'PINECONE',
                'query': user_query,
                'answer': final_answer,
                'matches_found': len(pinecone_matches),
                'subject_filter': subject_filter,
                'context_size': get_context_size(session_id),
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
        return jsonify({'success': False, 'error': 'An unexpected error occurred.'}), 500
