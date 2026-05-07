import mysql.connector

from config import DB_CONFIG


def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None


def execute_sql_query(connection, sql_query):
    cursor = connection.cursor(dictionary=True)
    cursor.execute(sql_query)
    raw_results = cursor.fetchall()
    cursor.close()
    connection.close()
    return raw_results
