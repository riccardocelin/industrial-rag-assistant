import yaml
from pathlib import Path
import requests

# load test query from config file
config_path = Path(__file__).parent / "config.test.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

test_query = config.get("test_query", "What is a DCS800 system?")
force_no_context_flag = config.get("force_no_context", False)
url = config.get("api_url", "http://localhost:8000/ask")

# send request to API
print(f"\nGenerating response {'without' if force_no_context_flag else 'with'} context...\n\n")
response = requests.post(
        url,
        json={
            "question": test_query,
            "force_no_context": force_no_context_flag
            }
    )

if response.status_code == 200:
    response_data = response.json()
    retrieved_docs = response_data.get("sources", [])

    print("\nRetrieved documents:")
    for i, doc_dict in enumerate(retrieved_docs):
        print(f"- doc{i + 1}. {doc_dict['text']}")

    print("\n\n\n######################## ANSWER ########################")    
    print("########################################################\n\n")   
    print(response_data.get("answer", "No answer in response"))
    print("\n\n\n######################## END ANSWER #####################")    
    print("########################################################")  

else:
    print(f"Error: {response.status_code} - {response.text}")
    retrieved_docs = []

