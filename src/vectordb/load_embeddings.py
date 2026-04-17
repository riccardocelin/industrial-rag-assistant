from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
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

def main():
    client = QdrantClient(host="localhost", port=6333)

    points = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            points.append(
                PointStruct(
                    id=record["metadata"]["chunk_id"],
                    vector=record["embedding"],
                    payload={
                        "text": record["text"],
                        "source": record["metadata"]["source_file"],
                        "page": record["metadata"]["page"],
                        "embedding_dim": record["embedding_metadata"]["embedding_dim"],
                        "embedding_hash": record["text_hash"]
                    }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"Inserted {len(points)} points into collection '{collection_name}'.")

if __name__ == "__main__":
    main()