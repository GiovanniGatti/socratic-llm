import argparse
import os
import pathlib

from tqdm import tqdm

from data import Scores, CrossValidationDataset, CrossValidation
from tools import ClientLLM, escape_template, safe_eval, JudgeLLM


def main(humans: pathlib.Path, eval_prompt: pathlib.Path, judge_llm: ClientLLM, output_path: pathlib.Path) -> None:
    with open(eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(humans) as f:
        human_eval = Scores.model_validate_json(f.read())

    client: ClientLLM = judge_llm
    cross_validations = CrossValidationDataset()
    for example in tqdm(human_eval.get_valid()):
        answer = example.output
        student = answer.split("Student")[0] if "Student" in answer else answer

        raw_evaluation, error, evaluation = safe_eval(
            client, judge_llm_prompt.format(conversation=example.prompt, answer=student)
        )

        cross_validation = CrossValidation(
            prompt=example.prompt, output=example.output, human=example.evaluation, gpt4o=evaluation
        )
        cross_validations.root.append(cross_validation)

    with open(output_path, "w") as f:
        f.write(cross_validations.model_dump_json(indent=2))

    os.chmod(output_path, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HUMAN_VS_GPT")
    parser.add_argument("--humans", required=True, type=pathlib.Path, help="Path to the evaluation by humans")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path, help="Path to the judge evaluation prompt")
    parser.add_argument(
        "--judge-llm", required=True, nargs=3, action=JudgeLLM,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct)."
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, help="Path to directory where to store judge LLM and human comparison data."
    )

    args = parser.parse_args()

    main(args.humans, args.eval_prompt, args.judge_llm, args.output_dir)
