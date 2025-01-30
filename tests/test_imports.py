# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_utils_imports():
    """Test imports in utils modules"""
    from src.utils import database_utils, db_selector
    assert hasattr(database_utils, 'get_db_connection')
    assert hasattr(database_utils, 'verify_table_structure')
    assert hasattr(db_selector, 'get_db_path')
    assert hasattr(db_selector, 'get_date_from_filename')
    assert hasattr(db_selector, 'get_experiment_time') 