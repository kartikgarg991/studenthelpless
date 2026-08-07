# StudentHelpless Project Context

This file captures the current project state after the modularization, health tracking, Redis session migration, and answer-formatting fixes. Use it as the starting context before planning future fixes, debugging, deployment work, or UI redesign.

## Current Status

- The project has been refactored from one large `app.py` and one large `index.html` into smaller files.
- Backend routes are still `/`, `/query`, and `/health`.
- Frontend behavior is still the same chat interface, only split into separate HTML, CSS, and JS files.
- Current conversation history is stored in Redis using `session_id`.
- `/health` reports local app-side Gemini/Pinecone usage without making extra Gemini/Pinecone calls.
- SQL empty-result hallucination is guarded: empty DB results now return a not-found message.
- DB answer formatting prompt is stricter so links and syllabus/unit data should be presented cleanly.
- Pinecone embedding uses `models/gemini-embedding-001` with dimensionality `768` by default.
- `.gitignore` now ignores secrets, Python cache files, virtualenvs, logs, and editor/OS files.
- `__pycache__` folders were created by Python syntax checking and are currently present because cleanup was not approved.

## Completed Work So Far

- Split backend into `routes/` and `services/`.
- Split frontend into `templates/index.html`, `static/css/style.css`, and `static/js/main.js`.
- Added `PROJECT_CONTEXT.md`.
- Added local usage tracking for Gemini and Pinecone calls.
- Enhanced `/health` with DB status, session stats, cache stats, and tracked usage.
- Replaced in-memory conversation history with Redis-backed history.
- Added Redis config: `REDIS_URL`, `SESSION_TTL_SECONDS`, and `MAX_HISTORY`.
- Added SQL empty-result guard.
- Tightened `format_final_answer()` prompt.
- Changed Pinecone query embedding model from old `models/text-embedding-004` to `models/gemini-embedding-001`.
- Added `redis==5.0.8` to `requirements.txt`.
- Updated `.gitignore`.

## File Structure

```text
studenthelpless/
  app.py
  config.py
  requirements.txt
  PROJECT_CONTEXT.md
  routes/
    __init__.py
    query.py
    health.py
  services/
    __init__.py
    cache_service.py
    db_fetch.py
    mistral_client.py
    llm_service.py
    pinecone_service.py
    session_store.py
    usage_tracker.py
  templates/
    index.html
  static/
    css/
      style.css
    js/
      main.js
```

## Backend Architecture

### `app.py`

Responsible for creating the Flask app.

- Creates `Flask(__name__)`.
- Enables CORS using `CORS(app)`.
- Registers the `/query` and `/health` blueprints.
- Defines `/`, which renders `templates/index.html`.
- Runs `initialize_cache()` before requests if cache is not initialized.
- Starts the app on `0.0.0.0` using `PORT` or `5000`.

### `config.py`

Loads environment variables using `python-dotenv`.

Current values expected from `.env`:

- `MISTRAL_API_KEY`
- `MISTRAL_CHAT_MODEL` optional, defaults to `mistral-small-latest`
- `MISTRAL_EMBEDDING_MODEL` optional, defaults to `mistral-embed`
- `MISTRAL_EMBEDDING_DIMENSIONALITY` optional, defaults to `768` (match your Pinecone index)
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `REDIS_URL`
- `SESSION_TTL_SECONDS` optional, defaults to `7200`
- `MAX_HISTORY` optional, defaults to `3`
- `DB_HOST`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `DB_PORT`

Important: the current `.env` contains live-looking secrets. They should eventually be rotated and kept out of shared project files.

### `routes/query.py`

Owns the `/query` POST endpoint.

Request body:

```json
{
  "query": "user question",
  "session_id": "browser-generated-session-id"
}
```

Current flow:

1. Read JSON body.
2. Extract `query`.
3. Extract `session_id`, defaulting to `"default"`.
4. Reject empty query with HTTP 400.
5. Cleanup expired Redis session bookkeeping.
6. Log the query and current session queue size.
7. Ask Gemini to classify the query as `SQL`, `PINECONE`, or `INVALID`.
8. Route the request based on classification.

Invalid path:

- Adds invalid response to session history.
- Returns HTTP 400 with a suggestion.

SQL path:

- Gemini generates SQL from the user query and cached schema.
- App connects to MySQL.
- App executes the generated SQL directly.
- If the SQL result is empty, the app returns a not-found message instead of asking Gemini to answer.
- If rows are returned, Gemini formats the raw database result into a final answer.
- Final answer is added to session history.
- Response includes `raw_data`.

Pinecone path:

- Gemini extracts a subject name using cached subjects and conversation context.
- Query is embedded using Gemini embedding model.
- Pinecone is searched with optional subject filter.
- Matches above score `0.55` are used.
- Gemini summarizes the matches.
- Final answer is added to session history.
- Response includes top match metadata.

### `routes/health.py`

Owns `/health`.

Current behavior:

- Actually checks MySQL by attempting `get_db_connection()`.
- Reports Gemini/Pinecone from local app-side usage tracking.
- Does not make extra Gemini or Pinecone calls during normal `/health`.
- Shows `unknown` for Gemini/Pinecone before any tracked call happens.
- Returns cache, session, and usage counts.

This should eventually support an optional deep health check such as `/health?deep=true`, but that has not been added yet.

## Services

### `services/cache_service.py`

Stores global app cache:

- `SUBJECTS_CACHE`
- `SCHEMA_CACHE`
- `CACHE_INITIALIZED`

On first request, it:

1. Connects to MySQL.
2. Loads distinct subject names from the `subjects` table.
3. Hardcodes schema metadata for:
   - `branches`
   - `subjects`
   - `syllabus`
   - `subject_pyqs`
4. Marks cache as initialized.

Current limitation:

- If DB subjects change after startup, cache does not refresh.
- Schema is hardcoded, not introspected from DB.

### `services/db_fetch.py`

Handles MySQL connection and SQL execution.

Functions:

- `get_db_connection()`
- `execute_sql_query(connection, sql_query)`

Current limitation:

- It executes whatever SQL string Gemini generated.
- No validation currently blocks writes, deletes, multi-statements, dangerous functions, or table access outside the expected schema.

### `services/mistral_client.py`

Wraps Mistral API calls:

- `chat_complete(prompt, system=None)` for chat completions
- `embed_text(text)` for vector embeddings (used for Pinecone queries)

### `services/llm_service.py`

Contains all LLM prompt logic (now using Mistral).

Functions:

- `classify_query_type(user_query, session_id)`
- `extract_subject_from_query(user_query, session_id)`
- `generate_pinecone_response(user_query, pinecone_matches, session_id)`
- `generate_sql_query(user_query, session_id)`
- `format_final_answer(user_query, raw_results, session_id)`

Current LLM responsibilities:

- Routing decision.
- Subject extraction.
- SQL generation.
- Final answer formatting.
- PYQ/Pinecone answer generation.

Current risk:

- The LLM has too much authority, especially because generated SQL is executed directly.

Current formatter guard:

- SQL empty results are blocked in `routes/query.py` before formatting.
- `format_final_answer()` is instructed to use only database returned data.
- It is instructed not to output raw JSON and to format URLs/syllabus fields cleanly.

### `services/pinecone_service.py`

Creates Pinecone client and index.

Search flow:

1. Embeds user query with configured Gemini embedding model, currently `models/gemini-embedding-001`.
2. Uses configured embedding dimensionality, currently `768`.
3. Builds optional filter:

```python
{"subject_name": {"$eq": subject_filter}}
```

4. Queries Pinecone with `top_k=10`.
5. Keeps matches with score greater than `0.55`.

Current limitation:

- Score threshold is hardcoded.
- No retry or timeout handling.
- If Pinecone/Gemini embedding fails, returns empty list and user gets a generic no-data answer.
- Embedding failures are tracked as Gemini failures; Pinecone index query failures are tracked as Pinecone failures.

### `services/session_store.py`

Current Redis session history implementation.

Uses Redis keys:

```text
chat:{session_id}
active_sessions
```

Current constants:

- `MAX_HISTORY = 3`
- `SESSION_TTL_SECONDS = 7200`

Current behavior:

- Each `session_id` has its own Redis chat key.
- Redis stores the last 3 user/assistant pairs.
- Redis key TTL is 2 hours by default.
- Full user and assistant messages are stored.
- LLM context still trims assistant previews to 150 characters.
- Context is formatted and sent into LLM prompts.
- If Redis is missing/down, history reads return empty context and writes are skipped.
- There is no in-memory fallback.

Important limitation:

- History persistence now depends on `REDIS_URL`.
- If Redis is unavailable, the app continues but conversation memory is unavailable.

### `services/usage_tracker.py`

Tracks local app-side usage for external AI/vector services without making extra API calls.

Currently tracks:

- `calls_today`
- `calls_last_minute`
- `last_call_at`
- `last_success_at`
- `last_error_at`
- `last_error_type`
- `last_error_message`
- `usable_likely`

Current tracked services:

- `gemini`
- `pinecone`

Important limitation:

- This does not know the official remaining quota from Google or Pinecone.
- It only records what this running app process has attempted.
- Counts reset if the Python process restarts.
- With multiple Gunicorn workers, each worker would have separate local usage counters.

## Frontend Architecture

### `templates/index.html`

Now contains only:

- HTML structure.
- Google font link.
- Marked.js CDN link.
- Link to `static/css/style.css`.
- Link to `static/js/main.js`.

Main DOM sections:

- Header.
- Stats badge.
- Clear chat button.
- Chat container.
- Empty state with quick action cards.
- Progress indicator.
- Query input.
- Send button.

### `static/css/style.css`

Contains the old inline CSS from `index.html`.

Current UI style:

- Dark theme.
- Glassmorphism.
- Animated mesh background.
- Large empty state.
- Quick action cards.
- Heavy gradients and emoji-based visual language.

Known concern:

- User dislikes current UI.
- Future redesign should make it calmer, cleaner, more academic, and less flashy.

### `static/js/main.js`

Contains the old inline JavaScript from `index.html`.

Current frontend flow:

1. Reads `session_id` from `localStorage`.
2. If no `session_id`, creates one with `crypto.randomUUID()`.
3. Computes API URL:
   - `http://localhost:5000/query` on localhost.
   - `${window.location.origin}/query` otherwise.
4. User sends a query by button click, Enter key, or quick action card.
5. User message is added to chat.
6. Input is disabled.
7. Progress indicator starts.
8. Frontend POSTs to backend with `query` and `session_id`.
9. Backend response is rendered into the chat.
10. Input is re-enabled.

Current frontend limitations:

- `clear chat` only clears browser UI, not backend session history.
- Markdown is rendered using `marked.parse(...)` into `innerHTML` without sanitization.
- Error messages are generic.
- Progress text is heuristic and not connected to actual backend steps.
- `messageCount` is local only and resets on page reload.

## End-to-End Request Flow

```text
Browser user input
  -> static/js/main.js handleSend()
  -> POST /query with query + session_id
  -> routes/query.py handle_query()
      -> services/session_store.py cleanup_sessions()
      -> services/llm_service.py classify_query_type()
          -> services/usage_tracker.py record_call("gemini")
      -> SQL path
          -> services/llm_service.py generate_sql_query()
          -> services/usage_tracker.py record_call("gemini")
          -> services/db_fetch.py get_db_connection()
          -> services/db_fetch.py execute_sql_query()
          -> services/llm_service.py format_final_answer()
          -> services/usage_tracker.py record_call("gemini")
          -> services/session_store.py add_to_history()
          -> JSON response
      -> PINECONE path
          -> services/llm_service.py extract_subject_from_query()
          -> services/usage_tracker.py record_call("gemini")
          -> services/pinecone_service.py search_pinecone()
          -> services/usage_tracker.py record_call("gemini") for embedding
          -> services/usage_tracker.py record_call("pinecone") for index query
          -> services/llm_service.py generate_pinecone_response()
          -> services/usage_tracker.py record_call("gemini")
          -> services/session_store.py add_to_history()
          -> JSON response
      -> INVALID path
          -> services/session_store.py add_to_history()
          -> HTTP 400 JSON response
  -> static/js/main.js addAssistantMessage()
  -> answer displayed in chat
```

## Known Problems To Fix Later

### Priority 1: SQL Safety

Current issue:

- Gemini-generated SQL is executed directly.

Needed fix:

- Validate that SQL is read-only.
- Allow only `SELECT`.
- Block dangerous keywords.
- Block multiple statements.
- Restrict allowed tables and columns.
- Consider using a query builder or constrained templates instead of free-form SQL generation.

### Priority 2: Secrets

Current issue:

- `.env` contains live-looking API keys and DB credentials.

Needed fix:

- Rotate credentials.
- Keep `.env` local only.
- Provide `.env.example` with placeholder values.

### Priority 3: Session Store

Current status:

- Session history has been moved to Redis.
- Redis failures are treated as empty history, not fatal query failures.

Possible future improvement:

- Add clear-session support.
- Confirm Render Redis/Upstash connection settings before deployment.

### Priority 4: Clear Chat Backend Sync

Current issue:

- Frontend clear button only clears UI.

Needed fix:

- Add `/clear-session`.
- Frontend should call it with `session_id`.
- Backend should delete that session's conversation history.

### Priority 5: Random Failure After 1-2 Messages

Possible causes:

- LLM classification drift from conversation context.
- Bad SQL generated after follow-up question.
- DB connection/query failure.
- Pinecone or Gemini timeout.
- Session context issue.
- Frontend receives HTTP 500 and hides details behind generic error.

Needed debugging:

- Add structured logs per request.
- Log query type, session id prefix, branch taken, and error category.
- Do not log secrets.
- Consider returning safe error codes like `DB_QUERY_FAILED`, `LLM_CLASSIFY_FAILED`, `PINECONE_FAILED`.

### Priority 6: Markdown Rendering Safety

Current issue:

- Frontend does `answer.innerHTML = marked.parse(...)`.

Needed fix:

- Sanitize rendered HTML with a library such as DOMPurify.
- Or render trusted markdown in a safer restricted mode.

### Priority 7: Real Health Checks

Current issue:

- `/health` now reports local app-side Gemini/Pinecone tracking, but it does not perform live checks.
- It cannot report official remaining quota.

Potential future fix:

- Add optional `/health?deep=true`.
- Deep mode can spend one tiny Gemini call intentionally.
- Deep mode can call Pinecone `describe_index_stats()` intentionally.
- Keep normal `/health` free of extra Gemini/Pinecone hits.

### Priority 8: UI Redesign

Current issue:

- User strongly dislikes current UI.
- It is flashy, emoji-heavy, and feels like a demo instead of a serious study tool.

Suggested direction:

- Clean academic chat workspace.
- Less animation and glow.
- Compact header.
- Useful sidebar for history/subjects later.
- Small quick prompt chips instead of large cards.
- Better answer formatting.
- Proper icons instead of emoji-heavy controls.

## Suggested Next Work Order

1. Confirm current modularized app still runs.
2. Use `/health` after test queries to inspect Gemini/Pinecone local usage and last errors.
3. Add SQL safety validation.
4. Add logging for the random 1-2 message failure.
5. Add `/clear-session`.
6. Add markdown sanitization.
7. Add optional `/health?deep=true`.
8. Retest Pinecone semantic search after API quota is available.
9. Redesign UI.

## Current Verification Already Done

- Python syntax check passed for the refactored backend modules.
- JavaScript syntax check passed for `static/js/main.js`.
- Python syntax check passed after adding usage tracking.
- Python syntax check passed after replacing in-memory sessions with Redis.
- No application behavior test was run against live Gemini, MySQL, or Pinecone during the refactor.

## Important Notes For Future Work

- Do not assume the refactor fixed security or session durability. It only made the project cleaner.
- Keep future changes small and isolated.
- Before fixing the random runtime issue, add observability first so failures become visible.
- `session_store.py` now owns Redis-backed history.
- `db_fetch.py` is the best place to introduce SQL validation later.
- `static/css/style.css` and `static/js/main.js` are ready for UI redesign work without touching backend logic.
