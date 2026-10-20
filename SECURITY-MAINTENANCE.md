# Dependency maintenance

The served Streamlit, Gradio and Shiny clients use the pinned Python 3.11/3.12 runtime in `requirements.txt`. Install with `python -m pip install -r requirements.txt`. The separate historical YouTube/transcription ingestion scripts require additional external tools and are not the served runtime. Tests mock OpenAI/HTTP and check initial Streamlit startup.

Direct requirements are in `requirements.in`; regenerate the complete lock with `uv pip compile requirements.in --python-version 3.11 -o requirements.txt`. CI runs offline compatibility tests and audits every pinned dependency with pip-audit. A passing audit means no known advisories in that resolved graph at scan time.
