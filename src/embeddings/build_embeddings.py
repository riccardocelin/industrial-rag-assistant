import json
import os
import hashlib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Config loading
# =========================

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# I/O
# =========================

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Utils
# =========================

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def batchify(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# =========================
# Main logic
# =========================

def main():

    PRJ_ROOT = Path(__file__).parent.parent.parent
    file_dir = Path(__file__).parent

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL")


    config = load_config(Path(file_dir / "config.embeddings.yaml"))

    input_path = Path(PRJ_ROOT / config["input"]["input_dir"] / config["input"]["chunks_file"])
    output_path = Path(PRJ_ROOT / config["output"]["output_dir"] / config["output"]["chunks_embeddings_file"])
    batch_size = config["embeddings"]["batch_size"]
    skip_existing = config["embeddings"]["skip_existing"]

    rows = load_jsonl(input_path)

    # Skip computation if already done
    if skip_existing and output_path.exists():
        print("Embeddings already computed. Skipping.")
        return

    client = OpenAI(api_key=api_key)

    texts = []
    indices = []

    for i, row in enumerate(rows):
        text = row.get("text", "").strip()

        if not text:
            continue

        texts.append(text)
        indices.append(i)

    print(f"Total chunks to embed: {len(texts)}")

    for batch_idx, batch in enumerate(batchify(list(zip(indices, texts)), batch_size)):

        idxs = [i for i, _ in batch]
        batch_texts = [t for _, t in batch]

        response = client.embeddings.create(
            model=embedding_model,
            input=batch_texts
        )

        embeddings = [item.embedding for item in response.data]

        for i, text, emb in zip(idxs, batch_texts, embeddings):
            rows[i]["embedding"] = emb
            rows[i]["text_hash"] = compute_hash(text)
            rows[i]["embedding_metadata"] = {
                "model": embedding_model,
                "embedding_dim": len(emb)
                }

        print(f"Processed batch {batch_idx + 1}")

    save_jsonl(rows, output_path)

    print("Done. Embeddings saved.")


if __name__ == "__main__":
    main()