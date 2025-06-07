#!/usr/bin/env python3
"""
Isolated test runner that bypasses problematic imports.
"""

import os
import sys

# Set environment to indicate testing mode
os.environ['PYTEST_CURRENT_TEST'] = 'test'

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Run pytest with the specified arguments
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(sys.argv[1:]))
