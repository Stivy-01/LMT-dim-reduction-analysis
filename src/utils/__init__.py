# -*- coding: utf-8 -*-
"""
Utility functions for the LMT Analysis package.
"""

import os
import sys
from pathlib import Path

def add_src_to_path():
    """Add the src directory to the Python path for direct script execution."""
    current_file = Path(os.path.abspath(__file__))
    src_dir = current_file.parent.parent
    project_root = src_dir.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Export the utility function
__all__ = ['add_src_to_path']

# Add other utility imports
from .database_utils import (
    create_backup,
    get_db_connection,
    verify_table_structure
)

from .db_selector import (
    get_db_path,
    get_date_from_filename,
    get_experiment_time
)

# Update exports
__all__ += [
    # database_utils
    'create_backup',
    'get_db_connection',
    'verify_table_structure',
    
    # db_selector
    'get_db_path',
    'get_date_from_filename',
    'get_experiment_time'
]
