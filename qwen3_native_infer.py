import argparse
import json
import math
import random
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 native zero-shot ranking inference on new_data.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B", type=str)
    parser.add_argument("--data_dir", default="new_data", type=str)
    parser.add_argument("--dataset_prefix", default="Baby_Products", type=str, choices=["Baby_Products", "Video_Games"])
    parser.add_argument("--seq_max_len", default=10, type=int)
    parser.add_argument("--num_negatives", default=1000, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument("--max_users", default=0, type=int, help="0 means all test users.")
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def load_item_map(data_dir: str, dataset_prefix: str) -> Dict[int, str]:
    item_map_path = f"{data_dir}/{dataset_prefix}_i_map.tsv"
    item_id2name = {}
    with open(item_map_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            original, item_id = line.rstrip("\n").split("\t")
            item_id2name[int(item_id)] = original.strip()
    return item_id2name


def load_test_rows(data_dir: str, dataset_prefix: str) -> List[Tuple[int, List[int]]]:
    rows = []
    test_path = f"{data_dir}/{dataset_prefix}_user_items_negs_test.csv"
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            pos = [int(x) for x in parts[1].split(",") if x]
            if len(pos) < 2:
                continue
            rows.append((user_id, pos))
    return rows


def sample_candidates(all_item_ids: List[int], history: List[int], target: int, num_negatives: int, rng: random.Random):
    banned = set(history)
    banned.add(target)
    pool = [x for x in all_item_ids if x not in banned]
    if len(pool) < num_negatives:
        raise ValueError(f"Not enough negatives for strict sampling: need={num_negatives}, available={len(pool)}")
    sampled = rng.sample(pool, num_negatives)
    candidates = sampled + [target]
    rng.shuffle(candidates)
    if len(candidates) != num_negatives + 1:
        raise ValueError(f"Invalid candidate size: expected={num_negatives + 1}, got={len(candidates)}")
    return candidates


def extract_ranked_ids(text: str, candidate_set: set) -> List[int]:
    ids = [int(x) for x in re.findall(r"\d+", text)]
    ranked = []
    used = set()
    for x in ids:
        if x in candidate_set and x not in used:
            ranked.append(x)
            used.add(x)
    return ranked


def calc_metrics(ranks: List[int]) -> Dict[str, float]:
    if not ranks:
        return {"hr@10": 0.0, "hr@20": 0.0, "hr@40": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0, "ndcg@40": 0.0}
    out = {}
    for k in [10, 20, 40]:
        hit = 0.0
        ndcg = 0.0
        for r in ranks:
            if r <= k:
                hit += 1.0
                ndcg += 1.0 / math.log2(r + 1.0)
        out[f"hr@{k}"] = hit / len(ranks)
        out[f"ndcg@{k}"] = ndcg / len(ranks)
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"[Init] Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print("[Init] Model ready.")

    item_id2name = load_item_map(args.data_dir, args.dataset_prefix)
    all_item_ids = sorted(item_id2name.keys())
    test_rows = load_test_rows(args.data_dir, args.dataset_prefix)
    if args.max_users > 0:
        test_rows = test_rows[:args.max_users]
    print(f"[Data] dataset={args.dataset_prefix} users={len(test_rows)} items={len(all_item_ids)}")

    ranks = []
    for idx, (user_id, pos) in enumerate(test_rows, start=1):
        target = pos[-1]
        history = pos[:-1][-args.seq_max_len:]
        candidates = sample_candidates(all_item_ids, history, target, args.num_negatives, rng)
        candidate_set = set(candidates)

        history_str = ", ".join([f"{x}" for x in history])
        candidate_str = ", ".join([str(x) for x in candidates])
        prompt = (
            "You are a recommendation ranking model.\n"
            f"Given user interaction history item_ids: [{history_str}]\n"
            f"Candidate item_ids (exactly {len(candidates)} ids): [{candidate_str}]\n"
            "Task: rank ALL candidate item_ids from most likely next-item to least likely.\n"
            "Output format MUST be JSON: {\"ranked_item_ids\": [id1, id2, ...]} with every candidate exactly once."
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            think_end = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            think_end = 0
        thinking_content = tokenizer.decode(output_ids[:think_end], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[think_end:], skip_special_tokens=True).strip("\n")

        ranked_ids = []
        try:
            parsed = json.loads(content)
            raw_ranked = parsed.get("ranked_item_ids", [])
            ranked_ids = [int(x) for x in raw_ranked if int(x) in candidate_set]
        except Exception:
            ranked_ids = extract_ranked_ids(content, candidate_set)

        used = set(ranked_ids)
        missing = [x for x in candidates if x not in used]
        ranked_ids.extend(missing)

        try:
            rank = ranked_ids.index(target) + 1
        except ValueError:
            rank = len(ranked_ids) + 1
        ranks.append(rank)

        avg = calc_metrics(ranks)
        print(f"[User {idx}/{len(test_rows)}] user_id={user_id} target={target} rank={rank}/{len(ranked_ids)}")
        if args.enable_thinking:
            print(f"[Thinking] {thinking_content[:300]}")
        print(f"[Output] {content[:300]}")
        print(
            f"[RunningAvg] hr@10={avg['hr@10']:.6f} hr@20={avg['hr@20']:.6f} hr@40={avg['hr@40']:.6f} "
            f"ndcg@10={avg['ndcg@10']:.6f} ndcg@20={avg['ndcg@20']:.6f} ndcg@40={avg['ndcg@40']:.6f}"
        )

    final_avg = calc_metrics(ranks)
    print("[FinalMetrics]", final_avg)


if __name__ == "__main__":
    main()
