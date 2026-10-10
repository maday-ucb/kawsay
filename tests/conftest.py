"""Pytest configuration for Kawsay tests."""
import sys
from pathlib import Path

# Ensure the project root is on the path so imports work from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
