import sqlite3
import yaml
import pandas as pd
from pathlib import Path

def normalize_spaces(name):
    """Convert spaces and hyphens to underscores while preserving case"""
    return name.replace(' ', '_').replace('-', '_')

def normalize_sequence_name(name):
    """Normalize sequence names by converting ' - ' to '___'"""
    return name.replace(' - ', '___')

# Load YAML taxonomy
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# Extract behaviors from YAML
yaml_behaviors = set()
behavior_variants = {}  # Track variants of each behavior

for event, details in settings['event_taxonomy']['events'].items():
    normalized_event = normalize_spaces(event)
    yaml_behaviors.add(normalized_event)
    
    if 'variants' in details:
        behavior_variants[normalized_event] = set()
        for variant in details['variants']:
            variant_name = normalize_spaces(variant['name'])
            variant_name = normalize_sequence_name(variant_name)
            yaml_behaviors.add(variant_name)
            behavior_variants[normalized_event].add(variant_name)

print("Behaviors in YAML:")
for behavior in sorted(yaml_behaviors):
    print(f"- {behavior}")

# Connect to database and get column names
try:
    conn = sqlite3.connect(r"C:\Users\andre\Desktop\exps\lda_database.sqlite")
    cursor = conn.cursor()
    
    # Get column names from merged_data
    cursor.execute("PRAGMA table_info(merged_data)")
    columns = [row[1] for row in cursor.fetchall()]
    
    print("\nAll columns in Database:")
    for col in columns:
        print(f"- {col}")
    
    # Process column names to extract behaviors
    db_behaviors = set()
    for col in columns:
        # Remove _active and _passive suffixes
        base_name = col.replace('_active', '').replace('_passive', '')
        if base_name not in ['mouse_id', 'interval_start', 'date']:
            db_behaviors.add(base_name)  # Add behavior name as is
    
    print("\nBehaviors in Database (after filtering):")
    for behavior in sorted(db_behaviors):
        print(f"- {behavior}")
    
    # Add parent behavior if any of its variants are present
    for parent, variants in behavior_variants.items():
        if any(variant in db_behaviors for variant in variants):
            db_behaviors.add(parent)
        
    # Compare sets
    print("\nAnalysis:")
    print("Behaviors only in YAML:")
    for b in sorted(yaml_behaviors - db_behaviors):
        print(f"- {b}")
    
    print("\nBehaviors only in Database:")
    for b in sorted(db_behaviors - yaml_behaviors):
        print(f"- {b}")
    
    print("\nBehaviors in both:")
    for b in sorted(yaml_behaviors & db_behaviors):
        print(f"- {b}")
    
    # Print counts
    print(f"\nTotal behaviors in YAML: {len(yaml_behaviors)}")
    print(f"Total behaviors in Database: {len(db_behaviors)}")
    print(f"Behaviors in both: {len(yaml_behaviors & db_behaviors)}")
        
except sqlite3.Error as e:
    print(f"Database error: {str(e)}")
finally:
    if 'conn' in locals():
        conn.close() 