import argparse
import os
import pathlib
from typing import Tuple, List

import torch
from openai import OpenAI
from tqdm import tqdm
from tqdm.contrib import tzip
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import Dataset, TrainDataset, DPOExample, DPOEvaluation
from tools import escape_template, safe_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GEN-TRAIN-DATASET")

    parser.add_argument("--input", required=True, type=pathlib.Path, help="Path to seed dataset")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path, help="Path to the inference prompt")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path, help="Path to the judge evaluation prompt")
    parser.add_argument("--instruct-model", required=True, type=str, help="HF path to instruct model")
    parser.add_argument("--openai-api-key", required=True, type=str, help="Open AI api key")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path where to store training dataset")

    args = parser.parse_args()

    torch.manual_seed(0)

    with open(args.eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(args.inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = escape_template(file.read())

    with open(args.input) as f:
        eval_prompts = Dataset.model_validate_json(f.read())

    model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model, trust_remote_code=True)

    answers: List[Tuple[str, str]] = []
    for prompt in tqdm(eval_prompts):
        _prompt = inference_prompt_template.format(input=prompt)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": _prompt}, ], tokenize=False, add_generation_prompt=True
        )
        encoded_inputs = tokenizer([formatted, ], return_tensors="pt").to("cuda")

        generate_kwargs = dict(encoded_inputs, max_new_tokens=250, do_sample=True, temperature=1.2)

        output = model.generate(**generate_kwargs)
        response_a = tokenizer.decode(output[0], skip_special_tokens=True)[len(_prompt) + 1:]

        output = model.generate(**generate_kwargs)
        response_b = tokenizer.decode(output[0], skip_special_tokens=True)[len(_prompt) + 1:]
        answers.append((response_a, response_b))

    client = OpenAI(api_key=args.openai_api_key)
    train_dataset = TrainDataset()
    for prompt, (answer_a, answer_b) in tzip(eval_prompts, answers):
        _answer_a = answer_a.split("Student")[0] if "Student" in answer_a else answer_a
        _answer_b = answer_b.split("Student")[0] if "Student" in answer_b else answer_b

        raw_evaluation_a, error_a, evaluation_a = safe_eval(
            client, judge_llm_prompt.format(conversation=prompt, answer=_answer_a)
        )

        raw_evaluation_b, error_b, evaluation_b = safe_eval(
            client, judge_llm_prompt.format(conversation=prompt, answer=_answer_b)
        )

        train_dataset.root.append(
            DPOExample(
                prompt=prompt,
                a=DPOEvaluation(
                    output=_answer_a, raw_evaluation=raw_evaluation_a, evaluation_error=error_a, evaluation=evaluation_a
                ),
                b=DPOEvaluation(
                    output=_answer_b, raw_evaluation=raw_evaluation_b, evaluation_error=error_b, evaluation=evaluation_b
                ),
            )
        )

    with open(args.output, "w") as f:
        f.write(train_dataset.model_dump_json(indent=2))

    os.chmod(args.output, 0o755)
