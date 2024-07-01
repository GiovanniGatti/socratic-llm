import argparse
import os
import pathlib

import torch
from openai import OpenAI
from peft import AutoPeftModelForCausalLM, PeftConfig
from pydantic import ValidationError
from pydantic_core import from_json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import Dataset, Evaluation, Example, Scores
from tools import escape_template

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
        p = inference_prompt_template.format(input=prompt)
        encoded_inputs = tokenizer.encode(p, return_tensors="pt").to("cuda")

        generate_kwargs = dict(
            input_ids=encoded_inputs,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_new_tokens=200,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        output = model.generate(**generate_kwargs)
        response = tokenizer.decode(output[0])[len(p):]
        answers.append(response)

    client = OpenAI(api_key=args.openai_api_key)
    scores = Scores()
    for prompt, answer in tqdm(zip(eval_prompts, answers)):
        student = answer.split("Student")[0] if "Student" in answer else answer
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": judge_llm_prompt.format(conversation=prompt, answer=student)},
            ],
            model="gpt-4o",
            temperature=0.2,
            seed=0
        )
        content = chat_completion.choices[0].message.content
        try:
            evaluation = Evaluation.model_validate(from_json(content, allow_partial=True))
        except ValidationError | ValueError as e:
            print("Evaluation error " + str(e))
            continue
        scores.root.append(Example(prompt=prompt, output=answer, evaluation=evaluation))

    with open(args.output, "w") as f:
        f.write(scores.model_dump_json(indent=2))

    os.chmod(args.output, 0o755)
