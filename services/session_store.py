import json
import time

import redis

from config import MAX_HISTORY, REDIS_URL, SESSION_TTL_SECONDS


ACTIVE_SESSIONS_KEY = "active_sessions"
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not REDIS_URL:
        raise ValueError("REDIS_URL is missing")
    _client = redis.from_url(REDIS_URL, decode_responses=True, max_connections=50)
    return _client


def _chat_key(session_id):
    return f"chat:{session_id}"


def _normalize_history(history):
    if not isinstance(history, list):
        return []

    normalized = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        user = turn.get("user")
        assistant = turn.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            continue
        normalized.append({"user": user, "assistant": assistant})
    return normalized[-MAX_HISTORY:]


def _read_history(session_id):
    client = _get_client()
    raw_history = client.get(_chat_key(session_id))
    if not raw_history:
        return []
    try:
        history = json.loads(raw_history)
    except json.JSONDecodeError:
        return []
    return _normalize_history(history)


def _save_history(session_id, history):
    client = _get_client()
    trimmed_history = _normalize_history(history)
    client.set(_chat_key(session_id), json.dumps(trimmed_history), ex=SESSION_TTL_SECONDS)
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    client.zadd(ACTIVE_SESSIONS_KEY, {session_id: expires_at})
    return trimmed_history


def add_to_history(session_id, user_msg, assistant_msg):
    try:
        history = _read_history(session_id)
        history.append({'user': user_msg, 'assistant': assistant_msg})
        saved_history = _save_history(session_id, history)
        print(f"📝 Session {session_id[:8]}... queue size: {len(saved_history)}/{MAX_HISTORY}")
    except Exception as e:
        print(f"❌ Redis history write skipped: {e}")


def get_context_for_llm(session_id):
    try:
        queue = _read_history(session_id)
    except Exception as e:
        print(f"❌ Redis history read skipped: {e}")
        queue = []

    if not queue:
        return "No previous conversation context."

    context = "RECENT CONVERSATION HISTORY:\n"
    for i, conv in enumerate(queue, 1):
        preview = conv['assistant'][:150] + "..." if len(conv['assistant']) > 150 else conv['assistant']
        context += f"\n[Turn {i}]\n"
        context += f"User: {conv['user']}\n"
        context += f"Assistant: {preview}\n"
    return context


def cleanup_sessions():
    try:
        client = _get_client()
        now = int(time.time())
        expired = client.zrangebyscore(ACTIVE_SESSIONS_KEY, "-inf", now)
        if not expired:
            return
        for session_id in expired:
            client.delete(_chat_key(session_id))
        client.zrem(ACTIVE_SESSIONS_KEY, *expired)
        print(f"🧹 Cleaned up {len(expired)} expired sessions")
    except Exception as e:
        print(f"❌ Redis cleanup skipped: {e}")


def get_context_size(session_id):
    try:
        return len(_read_history(session_id))
    except Exception as e:
        print(f"❌ Redis context size read skipped: {e}")
        return 0


def get_active_session_count():
    try:
        client = _get_client()
        now = int(time.time())
        client.zremrangebyscore(ACTIVE_SESSIONS_KEY, "-inf", now)
        return client.zcard(ACTIVE_SESSIONS_KEY)
    except Exception as e:
        print(f"❌ Redis active session count skipped: {e}")
        return 0
