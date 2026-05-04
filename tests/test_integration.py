import pytest

@pytest.mark.integration
def test_rag_api():
    import requests
    from app.core.settings import get_settings

    settings = get_settings()

    url = f"http://{settings.api_host}:{settings.api_port}{settings.api_ask_endpoint}"

    # send request to API
    response = requests.post(
            url,
            json={
                "question": "What is the maintenance schedule for the DSC800 system?",
                "force_no_context": False
                }
        )

    response_data = response.json()
    retrieved_docs = response_data.get("sources", [])
    answer = response_data.get("answer", [])

    assert response.status_code == 200
    assert isinstance(retrieved_docs, list) and len(retrieved_docs) > 0
    assert isinstance(answer, str) and len(answer) > 0
