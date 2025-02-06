import sqlite3
import pandas as pd
from pathlib import Path

def analyze_database(db_path, name):
    print(f"\n=== Analyzing {name} ===")
    conn = sqlite3.connect(db_path)
    
    # Get basic stats
    count = pd.read_sql('SELECT COUNT(*) as count FROM behavior_stats_intervals', conn).iloc[0]['count']
    print(f"Total rows: {count}")
    
    # Get interval distribution
    intervals = pd.read_sql('''
        SELECT interval_start, COUNT(*) as count, COUNT(DISTINCT mouse_id) as mice
        FROM behavior_stats_intervals 
        GROUP BY interval_start
        ORDER BY interval_start
    ''', conn)
    print("\nIntervals distribution:")
    print(intervals)
    
    # Get column info
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(behavior_stats_intervals)")
    columns = cursor.fetchall()
    print(f"\nNumber of columns: {len(columns)}")
    
    # Sample some actual data
    print("\nSample data (first row):")
    sample = pd.read_sql('SELECT * FROM behavior_stats_intervals LIMIT 1', conn)
    print(sample)
    
    conn.close()
    return intervals

def analyze_csv(csv_path):
    print(f"\n=== Analyzing CSV ===")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    if 'interval_start' in df.columns:
        intervals = df.groupby('interval_start').agg({
            'mouse_id': ['count', 'nunique']
        }).reset_index()
        intervals.columns = ['interval_start', 'count', 'mice']
        print("\nIntervals distribution:")
        print(intervals)
    
    print(f"\nNumber of columns: {len(df.columns)}")
    print("\nSample data (first row):")
    print(df.iloc[0])

def main():
    # Analyze source databases
    db1_intervals = analyze_database(
        r"C:\Users\andre\Desktop\exps\4wt 14_05_24.sqlite",
        "Database 1 (4wt)"
    )
    
    db2_intervals = analyze_database(
        r"C:\Users\andre\Desktop\exps\16p wt stress day 17_05_24.sqlite",
        "Database 2 (16p)"
    )
    
    # Analyze merged database
    merged_intervals = analyze_database(
        r"C:\Users\andre\Desktop\exps\merged_analysis.sqlite",
        "Merged Database"
    )
    
    # Analyze CSV
    csv_path = Path(r"C:\Users\andre\Desktop\LMT dim reduction toolkit\src\data\merged_analysis_behavior_stats_intervals.csv")
    if csv_path.exists():
        analyze_csv(csv_path)
    else:
        print("\nCSV file not found!")
    
    # Compare intervals
    print("\n=== Interval Analysis ===")
    all_intervals = pd.concat([
        db1_intervals.assign(source='db1'),
        db2_intervals.assign(source='db2'),
        merged_intervals.assign(source='merged')
    ])
    print("\nAll intervals from all sources:")
    print(all_intervals.sort_values(['interval_start', 'source']))

if __name__ == "__main__":
    main() 