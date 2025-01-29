# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import our modules
import pandas as pd
import sqlite3
from tkinter import filedialog
import tkinter as tk
from src.utils.db_selector import get_db_path
from src.utils.database_utils import get_db_connection

def conversion_to_csv(conn, db_path, table_name):
    """Convert specified table from database to CSV"""
    try:
        # Generate CSV path based on database path
        csv_path = os.path.splitext(db_path)[0] + ".csv"
        
        # Load data into pandas DataFrame
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Successfully created CSV at: {csv_path}")
        return True
    except Exception as e:
        print(f"❌ CSV conversion error: {str(e)}")
        return False

def get_columns(conn, db_path, table_name):
    """Fetch column names from a table in an attached database."""
    cursor = conn.cursor()
    columns = []
    try:
        cursor.execute("ATTACH DATABASE ? AS source_db", (db_path,))
        cursor.execute(f"PRAGMA source_db.table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Warning: Could not read columns from {db_path} - {str(e)}")
    finally:
        try:
            cursor.execute("DETACH DATABASE source_db")
        except:
            pass
    return columns

def verify_database(db_path, required_columns):
    """Verify if a database contains the required table and columns."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='behavior_hourly'")
        if not cursor.fetchone():
            return False
        
        # Check required columns
        cursor.execute("PRAGMA table_info(behavior_hourly)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        return all(col in existing_columns for col in required_columns)
        
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def main():
    required_columns = ['interval_start', 'mouse_id']
    table_name = 'behavior_hourly'
    
    try:
        # Select source databases using GUI
        print("Select source databases or folder containing databases...")
        db_paths = get_db_path()
        if not db_paths:
            raise ValueError("No databases selected")
        
        # Verify all selected databases
        valid_dbs = []
        for db_path in db_paths:
            if verify_database(db_path, required_columns):
                valid_dbs.append(db_path)
            else:
                print(f"Skipping invalid database: {Path(db_path).name}")
        
        if not valid_dbs:
            raise ValueError("No valid databases selected")
        
        # Select output path using GUI
        print("Select output location for merged database...")
        root = tk.Tk()
        root.withdraw()
        new_db_path = filedialog.asksaveasfilename(
            title="Save Merged Database As",
            defaultextension=".sqlite",
            filetypes=[("SQLite Databases", "*.sqlite"), ("All Files", "*.*")]
        )
        if not new_db_path:
            raise ValueError("No output location selected")
        
        # Create new database
        conn = sqlite3.connect(new_db_path)
        cursor = conn.cursor()
        
        # Get common columns across all databases
        common_cols = None
        for db_path in valid_dbs:
            cols = get_columns(conn, db_path, table_name)
            if common_cols is None:
                common_cols = set(cols)
            else:
                common_cols &= set(cols)
        
        # Ensure required columns are present
        common_cols = sorted(common_cols, 
                           key=lambda x: (required_columns.index(x) if x in required_columns else len(required_columns)))
        
        # Attach all valid databases
        attach_commands = []
        for i, db_path in enumerate(valid_dbs):
            alias = f"db{i+1}"
            cursor.execute(f"ATTACH DATABASE ? AS {alias}", (db_path,))
            attach_commands.append(f"SELECT {', '.join(common_cols)} FROM {alias}.{table_name}")
        
        # Create merged table
        union_query = " UNION ALL ".join(attach_commands)
        cursor.execute(f"""
            CREATE TABLE merged_data_hourly AS
            {union_query}
        """)
        
        conn.commit()
        print(f"\n✅ Successfully created merged database at: {new_db_path}")
        print(f"Merged {len(valid_dbs)} databases")
        print(f"Total records: {cursor.execute('SELECT COUNT(*) FROM merged_data_hourly').fetchone()[0]}")
        
        # Convert to CSV
        print("\nCreating merged CSV file...")
        if conversion_to_csv(conn, new_db_path, 'merged_data_hourly'):
            print("✅ Conversion to CSV completed successfully")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            # Cleanup attached databases
            try:
                for i in range(len(valid_dbs)):
                    cursor.execute(f"DETACH DATABASE db{i+1}")
            except:
                pass
            conn.close()

if __name__ == "__main__":
    main()