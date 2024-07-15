import argparse
import os
import pathlib

from tqdm import tqdm

from data import Dataset, Scores, Example
from tools import escape_template, safe_eval, JudgeLLM, ClientLLM


def main(
        dataset: pathlib.Path,
        inference_prompt: pathlib.Path,
        eval_prompt: pathlib.Path,
        judge_llm: ClientLLM,
        output_path: pathlib.Path
) -> None:
    with open(eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = file.read()

    with open(dataset) as f:
        eval_prompts = Dataset.model_validate_json(f.read())

    client: ClientLLM = judge_llm

    scores = Scores()
    for prompt in tqdm(eval_prompts):
        answer = client.chat([{"role": "user", "content": inference_prompt_template.format(input=prompt)}])

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

    with open(output_path, "w") as f:
        f.write(scores.model_dump_json(indent=2))

    os.chmod(output_path, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SELF-EVAL",
        description="Uses the Judge LLM (usually a power LLM) for creating Socratic interactions. Then, self evaluate "
                    "these interactions."
    )
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to evaluation datasets")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path,
                        help="Path to the inference prompt template")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument(
        "--judge-llm", required=True, nargs=3, action=JudgeLLM,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct)."
    )
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to self evaluation")
    args = parser.parse_args()

    main(args.input, args.inference_prompt, args.eval_prompt, args.judge_llm, args.output)
