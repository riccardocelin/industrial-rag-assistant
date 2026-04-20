from app.rag.rag_system import RAG
import yaml
from pathlib import Path

# load test query from config file
config_path = Path(__file__).parent / "config.test.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

test_query = config.get("test_query", "What is a DCS800 system?")
force_no_context_flag = config.get("force_no_context", False)

rag = RAG()

retrieved_docs = rag.retrieve(test_query)

print("\nRetrieved documents:")
for i, doc_dict in enumerate(retrieved_docs):
    print(f"- doc{i + 1}. {doc_dict["text"]}")

response = None
if force_no_context_flag:
    print("\nGenerating response without context...")
    response = rag.generate(test_query, retrieved_docs, force_no_context=force_no_context_flag)
else:
    print("\nGenerating response with retrieved context...")
    response = rag.generate(test_query, retrieved_docs, force_no_context=force_no_context_flag)

print(response)