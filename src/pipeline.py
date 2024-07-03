import argparse
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="PIPELINE")
    parser.add_argument("--openai-api-key", required=False, type=str, help="Open AI api key")
    parser.add_argument("--evaluation-dir", required=True, type=str, help="Path where to store assessments")
    parser.add_argument("--figures-dir", required=True, type=str, help="Path where to store figures")
    parser.add_argument("--dpo-dir", required=True, type=str,
                        help="Path where to store DPO training data and model weights")
    parser.add_argument("--instruct-model", default="microsoft/Phi-3-mini-128k-instruct", type=str,
                        help="HF name of instruct model to finetune")
    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    args = parser.parse_args()

    OPENAI_API_KEY = args.openai_api_key
    if OPENAI_API_KEY is None:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    if OPENAI_API_KEY is None:
        raise ValueError("Must provide OPENAI_API_KEY either through command line or environment variable")

    for dataset in ("mathdial", "tutorchat"):
        target_dir = Path(f"{args.dpo_dir}/{dataset}")
        target_dir.mkdir(0o755, exist_ok=True)
        if not args.use_cache or not Path(f"{args.dpo_dir}/{dataset}/train_dataset.json").exists():
            subprocess.run(["python", "-m", "gen_train_dataset",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--instruct-model", args.instruct_model,
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{args.dpo_dir}/{dataset}/train_dataset.json"])

    for dataset in ("mathdial", "debugging", "tutorchat"):
        if not args.use_cache or not Path(f"{args.evaluation_dir}/{dataset}_finetuned.json").exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--peft-adapter", "giovanni-gatti-pinheiro/socratic-llm",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{args.evaluation_dir}/{dataset}_finetuned.json"])

        if not args.use_cache or not Path(f"{args.evaluation_dir}/{dataset}_finetuned_with_mathdial.json").exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--peft-adapter", "giovanni-gatti-pinheiro/socratic-llm-mathdial",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{args.evaluation_dir}/{dataset}_finetuned_with_mathdial.json"])

        if not args.use_cache or not Path(f"{args.evaluation_dir}/{dataset}_base.json").exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--peft-adapter", "giovanni-gatti-pinheiro/socratic-llm",
                            "--without-lora-adapter",
                            "--output", f"{args.evaluation_dir}/{dataset}_base.json"])

        if not args.use_cache or not Path(f"{args.evaluation_dir}/{dataset}_gpt4o.json").exists():
            subprocess.run(["python", "-m", "eval_gpt_4o",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{args.evaluation_dir}/{dataset}_gpt4o.json"])

    if not args.use_cache or not len(list(Path(f"{args.figures_dir}").glob("fig[2-4].svg"))) > 0:
        subprocess.run(["python", "-m", "figures.fig2_4",
                        "--humans", f"./datasets/mathdial_human_eval.json",
                        "--eval-prompt", "./templates/judge_llm.txt",
                        "--openai-api-key", OPENAI_API_KEY,
                        "--output", f"{args.figures_dir}"])

    subprocess.run(["python", "-m", "figures.fig5_6",
                    "--mathdial-finetuned", f"{args.evaluation_dir}/mathdial_finetuned.json",
                    "--mathdial-base", f"{args.evaluation_dir}/mathdial_base.json",
                    "--mathdial-gpt4o", f"{args.evaluation_dir}/mathdial_gpt4o.json",
                    "--debugging-finetuned", f"{args.evaluation_dir}/debugging_finetuned.json",
                    "--debugging-base", f"{args.evaluation_dir}/debugging_base.json",
                    "--debugging-gpt4o", f"{args.evaluation_dir}/debugging_gpt4o.json",
                    "--tutorchat-finetuned", f"{args.evaluation_dir}/tutorchat_finetuned.json",
                    "--tutorchat-base", f"{args.evaluation_dir}/tutorchat_base.json",
                    "--tutorchat-gpt4o", f"{args.evaluation_dir}/tutorchat_gpt4o.json",
                    "--output-dir", f"{args.figures_dir}"])

    subprocess.run(["python", "-m", "figures.table",
                    "--mathdial-finetuned-with-tutorchat", f"{args.evaluation_dir}/mathdial_finetuned.json",
                    "--debugging-finetuned-with-tutorchat", f"{args.evaluation_dir}/debugging_finetuned.json",
                    "--tutorchat-finetuned-with-tutorchat", f"{args.evaluation_dir}/tutorchat_finetuned.json",
                    "--mathdial-finetuned-with-mathdial",
                    f"{args.evaluation_dir}/mathdial_finetuned_with_mathdial.json",
                    "--debugging-finetuned-with-mathdial",
                    f"{args.evaluation_dir}/debugging_finetuned_with_mathdial.json",
                    "--tutorchat-finetuned-with-mathdial",
                    f"{args.evaluation_dir}/tutorchat_finetuned_with_mathdial.json",
                    "--output-dir", f"{args.figures_dir}"])
