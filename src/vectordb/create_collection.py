from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from pathlib import Path
import json
import yaml

PRJ_ROOT = Path(__file__).parent.parent.parent
file_dir = Path(__file__).parent
config_path = file_dir / "config.vectordb.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

chunks_file = PRJ_ROOT / config.get("chunks_file", "data/processed/chunks_with_embeddings.jsonl")
collection_name = config.get("collection_name", "technical_docs")

def check_collection_exists(client, collection_name):
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

def check_embedding_dimension_from_chunks(chunks_file):
    # Implement logic to read the chunks file and determine the embedding dimension from embeddig metadata.
    with open(chunks_file, "r") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError("Chunks file is empty.")
        
        chunk = json.loads(first_line)
        embedding_dim = chunk.get("embedding_metadata").get("embedding_dim")
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("Invalid embedding dimension found in the chunk.")
        return embedding_dim

def main():

    embedding_dim = None
    try:
        embedding_dim = check_embedding_dimension_from_chunks(chunks_file)
        print(f"Detected embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"Error checking embedding dimension: {e}")
        return
    
    client = QdrantClient(host="localhost", port=6333)

    if not check_collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")

if __name__ == "__main__":
    main()