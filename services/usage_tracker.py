from collections import defaultdict, deque
from datetime import datetime, timezone


usage_state = defaultdict(lambda: {
    'calls_today': 0,
    'call_times': deque(),
    'current_day': None,
    'last_call_at': None,
    'last_success_at': None,
    'last_error_at': None,
    'last_error_type': None,
    'last_error_message': None,
})


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat() if dt else None


def _error_type(error):
    message = str(error).lower()
    if 'quota' in message or 'resource_exhausted' in message:
        return 'quota_exceeded'
    if 'rate' in message or '429' in message:
        return 'rate_limited'
    if 'api key' in message or 'permission' in message or 'unauthenticated' in message or '403' in message:
        return 'auth_failed'
    if 'timeout' in message or 'deadline' in message:
        return 'timeout'
    return error.__class__.__name__


def record_call(service_name, success=True, error=None):
    now = _now()
    state = usage_state[service_name]
    day = now.date().isoformat()

    if state['current_day'] != day:
        state['current_day'] = day
        state['calls_today'] = 0
        state['call_times'].clear()

    state['calls_today'] += 1
    state['last_call_at'] = now
    state['call_times'].append(now)

    one_minute_ago = now.timestamp() - 60
    while state['call_times'] and state['call_times'][0].timestamp() < one_minute_ago:
        state['call_times'].popleft()

    if success:
        state['last_success_at'] = now
        state['last_error_type'] = None
        state['last_error_message'] = None
    else:
        state['last_error_at'] = now
        state['last_error_type'] = _error_type(error) if error else 'unknown_error'
        state['last_error_message'] = str(error)[:300] if error else None


def get_usage_summary(service_name):
    state = usage_state[service_name]
    now = _now()
    one_minute_ago = now.timestamp() - 60
    while state['call_times'] and state['call_times'][0].timestamp() < one_minute_ago:
        state['call_times'].popleft()

    return {
        'calls_today': state['calls_today'],
        'calls_last_minute': len(state['call_times']),
        'last_call_at': _iso(state['last_call_at']),
        'last_success_at': _iso(state['last_success_at']),
        'last_error_at': _iso(state['last_error_at']),
        'last_error_type': state['last_error_type'],
        'last_error_message': state['last_error_message'],
        'usable_likely': state['last_error_type'] not in {'quota_exceeded', 'rate_limited', 'auth_failed'},
    }


def get_all_usage_summary():
    return {
        'gemini': get_usage_summary('gemini'),
        'pinecone': get_usage_summary('pinecone'),
    }
