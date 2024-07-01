import argparse
import json
import pathlib

import torch
from openai import OpenAI
from peft import AutoPeftModelForCausalLM, PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def compose_prompt(element):
    o = element['output']
    if "Student" in o:
        o = o.split("Student")[0]

    s = f"####Here is the conversation:#### \n  {element['prompt']}\n #####Here is the answer#### : \n {o}\n\n"

    return s


def calculate_score(eval):
    sum = 0
    if eval['questions'] == "Yes":
        sum += 0.25
    if eval['reveal_answer'] == "No":
        sum += 0.25
    sum += 0.25 / 5 * eval['on_topic']
    sum += 0.25 / 5 * eval['helpful']
    return sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GEN-FINETUNE")
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to evaluation datasets")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path,
                        help="Path to the inference prompt")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument("--openai-api-key", required=True, type=str, help="Open AI api key")
    parser.add_argument("--without-lora-adapter", action="store_true", help="Disable LoRA adapter (no finetuning)")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to GPT-4o eval")
    args = parser.parse_args()

    with open(args.eval_prompt, 'r', encoding='utf-8') as file:
        judge_llm_prompt = file.read()

    with open(args.inference_prompt, 'r', encoding='utf-8') as file:
        inference_prompt_template = file.read()

    with open(args.input) as f:
        eval_prompts = json.load(f)

    base_model = PeftConfig.from_pretrained("giovanni-gatti-pinheiro/socratic-llm").base_model_name_or_path

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
            "giovanni-gatti-pinheiro/socratic-llm",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained("giovanni-gatti-pinheiro/socratic-llm", trust_remote_code=True)

    evaluation = [{"prompt": item} for item in eval_prompts]

    for element in tqdm(evaluation):
        prompt = element["prompt"]

        p = inference_prompt_template.format(input=prompt)

        encoded_inputs = tokenizer.encode(p, return_tensors="pt").to("cuda")

        generate_kwargs = dict(
            input_ids=encoded_inputs,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_new_tokens=200,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        output = model.generate(**generate_kwargs)
        response = tokenizer.decode(output[0])[len(p):]
        element["output"] = response

    client = OpenAI(api_key=args.openai_api_key)
    for element in tqdm(evaluation):
        out = compose_prompt(element)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": judge_llm_prompt},
                {"role": "user", "content": out, }
            ],
            model="gpt-3.5-turbo",
            temperature=0.2,
        )
        res = json.loads(chat_completion.choices[0].message.content)

        score = calculate_score(res)
        element["score"] = score
        element["evaluation"] = res

    with open(args.output, 'w') as f:
        json.dump(evaluation, f, indent=2)
