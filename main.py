"""Process entrypoint for Streamlit Community Cloud.

Patches the standard-library sqlite3 module with pysqlite3 BEFORE any
Chroma import (Streamlit Cloud's bundled SQLite is too old for Chroma),
then runs the actual app.

Streamlit executes this file with `streamlit run`, so the app is loaded
with runpy instead of spawning another Streamlit server (that would
fail with "Runtime instance already exists").
"""
import runpy
import sys

try:  # pragma: no cover - environment-specific
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

runpy.run_path("app/streamlit_app.py", run_name="__main__")
