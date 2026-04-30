import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import requests


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(expected: str, predicted: str) -> float:
    exp_tokens = re.findall(r"\w+", normalize_text(expected))
    pred_tokens = re.findall(r"\w+", normalize_text(predicted))

    if not exp_tokens and not pred_tokens:
        return 1.0
    if not exp_tokens or not pred_tokens:
        return 0.0

    exp_counter = Counter(exp_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((exp_counter & pred_counter).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(exp_tokens) if exp_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_source_pairs(sources_from_api):
    """
    Best-effort parser for API "sources" field.
    Supports list items like:
      - {"source": "file.pdf", "page": 5}
      - {"metadata": {"source": "file.pdf", "page": 5}}
      - {"document": "file.pdf", "source_page": 5}
    """
    pairs = set()
    for item in sources_from_api or []:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}

        source = (
            item.get("source")
            or item.get("document")
            or item.get("filename")
            or metadata.get("source")
            or metadata.get("document")
            or metadata.get("filename")
        )
        page = (
            item.get("page")
            or item.get("source_page")
            or item.get("page_number")
            or metadata.get("page")
            or metadata.get("source_page")
            or metadata.get("page_number")
        )

        if source is not None:
            source = str(source)
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None

        if source is not None:
            pairs.add((source, page))

    return pairs


def evaluate_item(item, api_url: str, force_no_context: bool = False):
    question = item["question"]
    expected_answer = item["expected_answer"]
    expected_sources = item.get("sources", [])
    expected_pages = item.get("source_pages", [])

    payload = {
        "question": question,
        "force_no_context": force_no_context,
    }

    response = requests.post(api_url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    predicted_answer = data.get("answer", "") or ""
    retrieved_sources_raw = data.get("sources", []) or []

    expected_pairs = set(zip(expected_sources, expected_pages))
    retrieved_pairs = extract_source_pairs(retrieved_sources_raw)

    source_page_hit = 1 if any(pair in retrieved_pairs for pair in expected_pairs) else 0

    expected_source_set = set(expected_sources)
    retrieved_source_set = {src for src, _ in retrieved_pairs if src is not None}
    source_hit = 1 if expected_source_set.intersection(retrieved_source_set) else 0

    score_f1 = token_f1(expected_answer, predicted_answer)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
        "expected_pairs": sorted(expected_pairs),
        "retrieved_pairs": sorted(retrieved_pairs),
        "source_hit": source_hit,
        "source_page_hit": source_page_hit,
        "answer_token_f1": score_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and answer quality from a JSON QA dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to evaluation JSON file (list of question/answer/source objects).",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/ask",
        help="RAG API endpoint.",
    )
    parser.add_argument(
        "--force-no-context",
        action="store_true",
        help="Pass force_no_context=true to API payload.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only first N examples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save per-sample detailed results as JSON.",
    )

    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON list of evaluation items.")

    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    all_results = []
    for i, item in enumerate(dataset, 1):
        try:
            result = evaluate_item(item, args.api_url, args.force_no_context)
            all_results.append(result)
            print(
                f"[{i}/{len(dataset)}] F1={result['answer_token_f1']:.3f} "
                f"source_hit={result['source_hit']} source_page_hit={result['source_page_hit']}"
            )
        except Exception as e:
            print(f"[{i}/{len(dataset)}] ERROR: {e}")

    if not all_results:
        raise RuntimeError("No successful evaluations. Check API and dataset format.")

    avg_f1 = mean(r["answer_token_f1"] for r in all_results)
    source_hit_rate = mean(r["source_hit"] for r in all_results)
    source_page_hit_rate = mean(r["source_page_hit"] for r in all_results)

    summary = {
        "num_samples": len(all_results),
        "api_url": args.api_url,
        "force_no_context": args.force_no_context,
        "avg_answer_token_f1": avg_f1,
        "source_hit_rate": source_hit_rate,
        "source_page_hit_rate": source_page_hit_rate,
    }

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        payload = {
            "summary": summary,
            "results": all_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results written to: {args.output}")


if __name__ == "__main__":
    main()
