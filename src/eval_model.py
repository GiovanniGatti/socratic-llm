import argparse
import os
import pathlib

import torch
from openai import OpenAI
from peft import AutoPeftModelForCausalLM, PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import Dataset, Example, Scores
from tools import escape_template, safe_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GEN-FINETUNE")
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to evaluation datasets")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path,
                        help="Path to the inference prompt")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument("--openai-api-key", required=True, type=str, help="Open AI api key")
    parser.add_argument("--peft-adapter", required=True, type=str, help="HF model path")
    parser.add_argument("--without-lora-adapter", action="store_true", help="Disable LoRA adapter (no finetuning)")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to GPT-4o eval")
    args = parser.parse_args()

    with open(args.eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(args.inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = escape_template(file.read())

    with open(args.input) as f:
        eval_prompts = Dataset.model_validate_json(f.read())

    base_model = PeftConfig.from_pretrained(args.peft_adapter).base_model_name_or_path

    if args.without_lora_adapter:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.peft_adapter,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.peft_adapter, trust_remote_code=True)

    answers = []
    for prompt in tqdm(eval_prompts):
        _prompt = inference_prompt_template.format(input=prompt)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": _prompt}, ], tokenize=False, add_generation_prompt=True
        )
        encoded_inputs = tokenizer([formatted, ], return_tensors="pt").to("cuda")

        generate_kwargs = dict(encoded_inputs, max_new_tokens=250)

        output = model.generate(**generate_kwargs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)[len(_prompt) + 1:]
        answers.append(response)

    client = OpenAI(api_key=args.openai_api_key)
    scores = Scores()
    for prompt, answer in tqdm(zip(eval_prompts, answers)):
        student = answer.split("Student")[0] if "Student" in answer else answer

        raw_evaluation, error, evaluation = safe_eval(
            client, judge_llm_prompt.format(conversation=prompt, answer=student)
        )

        scores.root.append(
            Example(
                prompt=prompt,
                output=answer,
                raw_evaluation=raw_evaluation,
                evaluation_error=error,
                evaluation=evaluation
            )
        )

    with open(args.output, "w") as f:
        f.write(scores.model_dump_json(indent=2))

    os.chmod(args.output, 0o755)
