import argparse
import pathlib

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from data import TrainDataset
from tools import escape_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TRAIN")

    parser.add_argument("--input", required=True, type=pathlib.Path, help="Path to seed dataset")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path, help="Path to the inference prompt")
    parser.add_argument("--instruct-model", required=True, type=str, help="HF path to instruct model")
    parser.add_argument("--checkpoints-dir", required=True, type=pathlib.Path,
                        help="Path where to store training checkpoints")
    parser.add_argument("--model-dir", required=True, type=pathlib.Path, help="Path where to store finetuned model")

    args = parser.parse_args()

    with open(args.inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = escape_template(file.read())

    with open(args.input, "r") as f:
        dataset = TrainDataset.model_validate_json(f.read())

    tlr_dataset = Dataset.from_dict({
        "prompt": [
            inference_prompt_template.format(input=i.prompt) for i in dataset.get_eligible_for_training()
        ],
        "chosen": [
            i.chosen for i in dataset.get_eligible_for_training()
        ],
        "rejected": [
            i.rejected for i in dataset.get_eligible_for_training()
        ]
    })

    model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)

    training_args = DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=0.3,
        num_train_epochs=2,
        learning_rate=5e-5,
        save_total_limit=3,
        logging_steps=10,
        output_dir=args.checkpoints_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        bf16=True
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=tlr_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()

    output_dir = pathlib.Path(args.model_dir)
    dpo_trainer.save_model(output_dir)
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
