"""ResearchLens backend.

Loading .env here rather than in main.py is deliberate: several modules read
their configuration into module-level constants at import time, and this package
__init__ runs before any of them. Loading it later would leave RESEARCHLENS_MODEL
and friends documented in .env.example but silently ignored.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency; environment variables still work
    pass
else:
    load_dotenv(Path(__file__).parent.parent / ".env")
