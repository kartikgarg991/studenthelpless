from flask import Blueprint, jsonify

from services.cache_service import get_schema_cache, get_subjects_cache
from services.db_fetch import get_db_connection
from services.session_store import MAX_HISTORY, get_active_session_count
from services.usage_tracker import get_all_usage_summary


health_bp = Blueprint('health', __name__)


def _tracked_status(summary):
    if summary['last_error_type'] in {'quota_exceeded', 'rate_limited', 'auth_failed'}:
        return summary['last_error_type']
    if summary['last_success_at']:
        return 'active'
    if summary['last_error_at']:
        return 'error'
    return 'unknown'


@health_bp.route('/health', methods=['GET'])
def health():
    db_status = 'active' if get_db_connection() else 'inactive'
    usage = get_all_usage_summary()
    return jsonify({
        'status': 'healthy',
        'services': {
            'database': db_status,
            'pinecone': _tracked_status(usage['pinecone']),
            'mistral': _tracked_status(usage['mistral'])
        },
        'cache_stats': {
            'subjects_cached': len(get_subjects_cache()),
            'schema_cached': bool(get_schema_cache())
        },
        'session_stats': {
            'active_sessions': get_active_session_count(),
            'max_history': MAX_HISTORY
        },
        'usage': usage
    })
