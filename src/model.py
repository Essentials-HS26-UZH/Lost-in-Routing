"""

Run language CSVs through a model. 

Prerequisites: .env file with OPENROUTER_API_KEY set to your OpenRouter API key.

Usage: python src\model.py --which big|small --input [src\belebele_LAN_processed.csv ...] [--limit 20]

"""

import argparse
import csv
import hashlib
import json
import os
import re
 
from dotenv import dotenv_values
from openai import OpenAI
 
KEY = dotenv_values().get("OPENROUTER_API_KEY") or ""
 
# (base_url, model, api_key) - change the model names here (e.g. to a free model for a test)
SETTINGS = {
    "big": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.1-70b-instruct", KEY),
    "small": ("http://localhost:11434/v1", "llama3.1:8b", "ollama"),
}
 
 
def load_cache(path):
    """Saved answers, so every question is only paid for once."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    return {e["key"]: e for e in entries}
 
 
def ask(client, model, prompt):
    """Send one prompt to the model."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    return {
        "text": response.choices[0].message.content or "",
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
 
 
def get_answer(client, model, prompt, cache, cache_path):
    """Return the model's answer, from the cache if possible."""
    key = hashlib.sha256((model + prompt).encode("utf-8")).hexdigest()
    if key not in cache:
        cache[key] = {"key": key, "model": model, **ask(client, model, prompt)}
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    return cache[key]
 
 
def parse_answer(text):
    """The number inside the last <answer> tag, or None."""
    found = re.findall(r"<answer>\s*([1-4])\s*</answer>", text)
    return int(found[-1]) if found else None
 
 
def read_rows(path, limit):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[:limit]
 
 
def make_result(row, answer):
    pred = parse_answer(answer["text"])
    gold = int(row["target_answer"])
    return {
        "question_id": row["question_id"],
        "language": row["language"],
        "gold": gold,
        "pred": pred,
        "correct": int(pred == gold),
        "prompt_tokens": answer["prompt_tokens"],
        "completion_tokens": answer["completion_tokens"],
        "raw_output": answer["text"],
    }
 
 
def save_results(results, which, path):
    """Saved next to this script (src/), e.g. oracle_en_big.csv."""
    lang = os.path.basename(path).split("_")[1]  # belebele_en_processed.csv -> en
    folder = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(folder, f"oracle_{lang}_{which}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return out_path
 
 
def run_file(path, which, limit, client, model, cache, cache_path):
    rows = read_rows(path, limit)
    results = []
    for i, row in enumerate(rows, 1):
        answer = get_answer(client, model, row["formatted_prompt"], cache, cache_path)
        results.append(make_result(row, answer))
        print(f"{path}: {i}/{len(rows)}", end="\r")
 
    out_path = save_results(results, which, path)
    accuracy = sum(r["correct"] for r in results) / len(results)
    tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in results)
    print(f"\n{out_path}: accuracy {accuracy:.1%}, {tokens} tokens")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["big", "small"], required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
 
    base_url, model, api_key = SETTINGS[args.which]
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)
    cache_path = f"cache_{args.which}.jsonl"
    cache = load_cache(cache_path)
 
    for path in args.input:
        run_file(path, args.which, args.limit, client, model, cache, cache_path)
 
 
if __name__ == "__main__":
    main()