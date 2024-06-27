import argparse
import json
import pathlib

from openai import OpenAI
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(prog="GEN-SOCRATIC-GPT-4o")
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to evaluation prompts")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path,
                        help="Path to the inference prompt template")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument("--openai-api-key", required=True, type=str, help="Open AI api key")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to GPT-4o eval")
    args = parser.parse_args()

    with open(args.eval_prompt, 'r', encoding='utf-8') as file:
        judge_llm_prompt = file.read()

    with open(args.inference_prompt, 'r', encoding='utf-8') as file:
        inference_prompt_template = file.read()

    with open(args.input) as f:
        eval_prompts = json.load(f)

    client = OpenAI(api_key=args.openai_api_key)

    evaluation = [{"prompt": item} for item in eval_prompts]

    for element in tqdm(evaluation):
        prompt = element["prompt"]

        p = inference_prompt_template.format(input=prompt)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": p},
            ],
            model="gpt-4o",
            temperature=0.2,
        )

        element["output"] = chat_completion.choices[0].message.content

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
