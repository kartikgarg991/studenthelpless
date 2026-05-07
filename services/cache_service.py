from services.db_fetch import get_db_connection


SUBJECTS_CACHE = []
SCHEMA_CACHE = {}
CACHE_INITIALIZED = False


def get_subjects_cache():
    return SUBJECTS_CACHE


def get_schema_cache():
    return SCHEMA_CACHE


def is_cache_initialized():
    return CACHE_INITIALIZED


def initialize_cache():
    global SUBJECTS_CACHE, SCHEMA_CACHE, CACHE_INITIALIZED
    if CACHE_INITIALIZED:
        return
    print("\n🔄 Initializing cache...")
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
    print("✅ Schema cached")
    CACHE_INITIALIZED = True
