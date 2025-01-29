import sqlite3
import pandas as pd

def create_backup(source_db, backup_db):
    """Create database backup"""
    con = sqlite3.connect(source_db)
    bck = sqlite3.connect(backup_db)
    con.backup(bck)
    bck.close()
    con.close()

def get_db_connection(db_path):
    """Create database connection"""
    return sqlite3.connect(db_path)

def verify_table_structure(conn):
    """Verify created tables have correct structure"""
    tables_to_verify = ['BEHAVIOR_STATS', 'MULTI_MOUSE_EVENTS']

    for table in tables_to_verify:
        try:
            pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
            print(f"✓ Table {table} exists and is accessible")
        except Exception as e:
            print(f"✗ Error accessing {table}: {str(e)}") 