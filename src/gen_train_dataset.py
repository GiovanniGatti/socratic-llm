import argparse
import os
import pathlib
from typing import List

import torch
from tqdm import tqdm
from tqdm.contrib import tzip
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import Dataset, TrainDataset, DPOExample, DPOEvaluation
from tools import escape_template, safe_eval, JudgeLLM, ClientLLM


def main(
        dataset: pathlib.Path,
        inference_prompt: pathlib.Path,
        eval_prompt: pathlib.Path,
        instruct_model: str,
        judge_llm: ClientLLM,
        output_path: pathlib.Path
) -> None:
    torch.manual_seed(0)

    with open(eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = escape_template(file.read())

    with open(dataset) as f:
        eval_prompts = Dataset.model_validate_json(f.read())

    model = AutoModelForCausalLM.from_pretrained(
        instruct_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(instruct_model, trust_remote_code=True)

    answers: List[List[str]] = []
    for prompt in tqdm(eval_prompts):
        _prompt = inference_prompt_template.format(input=prompt)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": _prompt}, ], tokenize=False, add_generation_prompt=True
        )
        encoded_inputs = tokenizer([formatted, ], return_tensors="pt").to("cuda")

        generate_kwargs = dict(encoded_inputs, max_new_tokens=400, do_sample=True, temperature=1.2)

        collected = []
        for _ in range(5):
            output = model.generate(**generate_kwargs)
            response = tokenizer.decode(output[0], skip_special_tokens=True)[len(_prompt) + 1:]
            collected.append(response)

        answers.append(collected)

    client = judge_llm
    train_dataset = TrainDataset()
    for prompt, collected in tzip(eval_prompts, answers):

        dpo_evaluations = []
        for _answer in collected:
            raw_evaluation, error, evaluation = safe_eval(
                client, judge_llm_prompt.format(conversation=prompt, answer=_answer)
            )
            dpo_evaluations.append(
                DPOEvaluation(
                    output=_answer, raw_evaluation=raw_evaluation, evaluation_error=error, evaluation=evaluation
                )
            )

        dpo_evaluations.sort(key=lambda e: e.summary_score(), reverse=True)

        chosen = dpo_evaluations[0]

        i = len(dpo_evaluations) - 1
        while i > 1 and dpo_evaluations[i].evaluation_error is not None:
            i -= 1
        rejected = dpo_evaluations[i]

        train_dataset.root.append(
            DPOExample(prompt=prompt, chosen_eval=chosen, rejected_eval=rejected, all_evaluations=dpo_evaluations)
        )

    with open(output_path, "w") as f:
        f.write(train_dataset.model_dump_json(indent=2))

    os.chmod(output_path, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GEN-TRAIN-DATASET",
        description="Generate the DPO training. First, using the instruct model, it creates several interactions."
                    "Then, it ranks these interactions using the JudgeLLM.",
    )

    parser.add_argument("--input", required=True, type=pathlib.Path, help="Path to seed dataset")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path, help="Path to the inference prompt")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path, help="Path to the judge evaluation prompt")
    parser.add_argument("--instruct-model", required=True, type=str, help="HF path to instruct model")
    parser.add_argument(
        "--judge-llm", required=True, nargs=3, action=JudgeLLM,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct)."
    )
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path where to store training dataset")

    args = parser.parse_args()

    main(args.input, args.inference_prompt, args.eval_prompt, args.intruct_model, args.judge_llm, args.output)
