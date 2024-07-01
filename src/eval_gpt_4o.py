import argparse
import os
import pathlib

from openai import OpenAI
from pydantic import ValidationError
from pydantic_core import from_json
from tqdm import tqdm

from data import Dataset, Evaluation, Scores, Example
from tools import escape_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GEN-SOCRATIC-GPT-4o")
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to evaluation datasets")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path,
                        help="Path to the inference prompt template")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument("--openai-api-key", required=True, type=str, help="Open AI api key")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to GPT-4o eval")
    args = parser.parse_args()

    with open(args.eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(args.inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = file.read()

    with open(args.input) as f:
        eval_prompts = Dataset.model_validate_json(f.read())

    client = OpenAI(api_key=args.openai_api_key)

    scores = Scores()
    for prompt in tqdm(eval_prompts):
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": inference_prompt_template.format(input=prompt)},
            ],
            model="gpt-4o",
            temperature=0.2,
            seed=0,
        )
        answer = chat_completion.choices[0].message.content

        student = answer.split("Student")[0] if "Student" in answer else answer
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": judge_llm_prompt.format(conversation=prompt, answer=student)},
            ],
            model="gpt-4o",
            temperature=0.2,
            seed=0,
        )
        content = chat_completion.choices[0].message.content
        try:
            evaluation = Evaluation.model_validate(from_json(content, allow_partial=True))
        except ValidationError as e:
            print("Evaluation error " + str(e))
            continue
        scores.root.append(Example(prompt=prompt, output=answer, evaluation=evaluation))

    with open(args.output, "w") as f:
        f.write(scores.model_dump_json(indent=2))

    os.chmod(args.output, 0o755)
