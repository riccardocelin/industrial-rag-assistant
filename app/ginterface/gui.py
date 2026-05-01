"""Streamlit GUI for querying the RAG API.

This interface mirrors the API usage in test/test_api.py by sending a POST
request with {"question": ..., "force_no_context": ...} to /ask.
"""

from __future__ import annotations

import requests
import streamlit as st

from app.core.settings import get_settings

settings = get_settings()
API_URL = settings.api_ask_endpoint_url
APP_NAME = settings.app_name

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Simple Streamlit interface for the /ask endpoint.")

question = st.text_area("Ask a question", height=160)
force_no_context = st.checkbox("Force no context", value=False)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        payload = {
            "question": question,
            "force_no_context": force_no_context,
        }

        with st.spinner("Generating answer..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()

                st.subheader("Answer")
                st.write(data.get("answer", ""))

                sources = data.get("sources", [])
                if not force_no_context and sources:
                    st.subheader("Sources")
                    for i, source in enumerate(sources, start=1):
                        with st.expander(f"Source {i}"):
                            st.json(source)
            except requests.exceptions.RequestException as exc:
                st.error(f"API request failed: {exc}")
