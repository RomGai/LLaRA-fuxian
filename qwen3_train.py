import argparse
import hashlib
import random
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-8B (LoRA) on new_data recommendation pairs.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B", type=str)
    parser.add_argument("--data_dir", default="new_data", type=str)
    parser.add_argument("--dataset_prefix", default="Baby_Products", choices=["Baby_Products", "Video_Games"], type=str)
    parser.add_argument("--seq_max_len", default=10, type=int)
    parser.add_argument("--val_user_ratio", default=0.05, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means all samples.")
    parser.add_argument("--output_dir", default="./checkpoints_qwen3", type=str)
    parser.add_argument("--num_train_epochs", default=1, type=float)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--logging_steps", default=1, type=int)
    parser.add_argument("--save_steps", default=200, type=int)
    parser.add_argument("--eval_steps", default=200, type=int)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    return parser.parse_args()


def load_item_map(data_dir: str, dataset_prefix: str) -> Dict[int, str]:
    path = f"{data_dir}/{dataset_prefix}_i_map.tsv"
    item_id2name = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            original, item_id = line.rstrip("\n").split("\t")
            item_id2name[int(item_id)] = original.strip()
    return item_id2name


def iter_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            user_id = int(parts[0])
            pos = [int(x) for x in parts[1].split(",") if x]
            if len(pos) < 2:
                continue
            yield user_id, pos


def in_val_split(user_id: int, seed: int, val_user_ratio: float) -> bool:
    digest = hashlib.md5(f"{user_id}_{seed}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return score < val_user_ratio


def leakage_check(data_dir: str, dataset_prefix: str):
    train_users = set([u for u, _ in iter_rows(f"{data_dir}/{dataset_prefix}_user_items_negs_train.csv")])
    test_users = set([u for u, _ in iter_rows(f"{data_dir}/{dataset_prefix}_user_items_negs_test.csv")])
    overlap = train_users & test_users
    if overlap:
        raise ValueError(f"Data leakage detected. overlap users: {len(overlap)}")
    print(f"[DataLeakageCheck] train_users={len(train_users)} test_users={len(test_users)} overlap=0")


def build_examples(args, item_id2name):
    train, val = [], []
    for user_id, pos in iter_rows(f"{args.data_dir}/{args.dataset_prefix}_user_items_negs_train.csv"):
        history = pos[:-1][-args.seq_max_len:]
        target = pos[-1]
        history_text = ", ".join([f"{i}({item_id2name.get(i, str(i))})" for i in history])
        target_text = f"{target}"
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a recommender system assistant. "
                    "Given user interaction history, predict the next item_id only.\n"
                    f"History: [{history_text}]"
                ),
            },
            {"role": "assistant", "content": target_text},
        ]
        example = {"messages": messages, "user_id": user_id}
        if in_val_split(user_id, args.seed, args.val_user_ratio):
            val.append(example)
        else:
            train.append(example)
    if args.max_samples > 0:
        train = train[:args.max_samples]
        val = val[: max(1, args.max_samples // 20)]
    return train, val


def preprocess_dataset(ds: Dataset, tokenizer, max_length: int):
    def _map_fn(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"][:]
        return tokens

    return ds.map(_map_fn, remove_columns=ds.column_names)


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        if loss is not None:
            print(f"[TrainStep] step={state.global_step} loss={loss} lr={lr}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    leakage_check(args.data_dir, args.dataset_prefix)
    item_id2name = load_item_map(args.data_dir, args.dataset_prefix)
    train_examples, val_examples = build_examples(args, item_id2name)
    print(f"[Data] train={len(train_examples)} val={len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = preprocess_dataset(Dataset.from_list(train_examples), tokenizer, args.max_length)
    val_ds = preprocess_dataset(Dataset.from_list(val_examples), tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[ProgressCallback()],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] Saved LoRA model/tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
