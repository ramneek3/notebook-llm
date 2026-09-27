"""Process entrypoint.

Patches the standard-library sqlite3 module with pysqlite3 BEFORE any
Chroma import. Streamlit Community Cloud ships a system SQLite that is
too old for Chroma; this shim is the standard fix.

Streamlit runs the app with:  streamlit run <file>
So this file simply delegates to app/streamlit_app.py.
"""
import sys

try:  # pragma: no cover - environment-specific
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from streamlit.web import cli as stcli  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app/streamlit_app.py", *sys.argv[1:]]
    sys.exit(stcli.main())
