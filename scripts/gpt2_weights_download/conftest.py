"""Pytest setup for the gpt2 weights download script.

Adds the script project root to ``sys.path`` so tests can import the
``gpt2_weights_download`` module by name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
