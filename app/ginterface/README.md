# Simple GUI for the RAG API

This folder contains a minimal Python GUI for querying the RAG API.

## What it does

- Provides one text box for the user question.
- Provides one text box to display the system answer.
- Sends a POST request to `http://localhost:8000/ask` using the same payload shape as `test/test_api.py`.

## Run

1. Start the API server (example):
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000
   ```
2. In another terminal, run the GUI:
   ```bash
   python -m streamlit run app/ginterface/gui.py
   ```

If your API URL is different, edit `API_URL` in `app/ginterface/gui.py`.
