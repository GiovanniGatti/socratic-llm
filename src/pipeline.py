import argparse
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="PIPELINE")
    parser.add_argument("--openai-api-key", required=False, type=str, help="Open AI api key")
    parser.add_argument("--evaluation-dir", required=True, type=str, help="Path where to store assessments")
    parser.add_argument("--figures-dir", required=True, type=str, help="Path where to store figures")
    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    args = parser.parse_args()

    OPENAI_API_KEY = args.openai_api_key
    if OPENAI_API_KEY is None:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    if OPENAI_API_KEY is None:
        raise ValueError("Must provide OPENAI_API_KEY either through command line or environment variable")

    for dataset in ("mathdial", "debugging", "tutorchat"):
        if args.use_cache and not Path(f"{args.evaluation_dir}/{dataset}_finetuned.json").exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{args.evaluation_dir}/{dataset}_finetuned.json"])

        if args.use_cache and not Path(f"{args.evaluation_dir}/{dataset}_base.json").exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--without-lora-adapter",
                            "--output", f"{args.evaluation_dir}/{dataset}_base.json"])

        if args.use_cache and not Path(f"{args.evaluation_dir}/{dataset}_gpt4o.json").exists():
            subprocess.run(["python", "-m", "eval_gpt_4o",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{args.evaluation_dir}/{dataset}_gpt4o.json"])

    if args.use_cache and not len(list(Path(f"{args.figures_dir}").glob("fig[2-4].svg"))) > 0:
        subprocess.run(["python", "-m", "figures.fig2_4",
                        "--humans", f"./datasets/mathdial_human_eval.json",
                        "--eval-prompt", "./templates/judge_llm.txt",
                        "--openai-api-key", OPENAI_API_KEY,
                        "--output", f"{args.figures_dir}"])

    subprocess.run(["python", "-m", "figures.fig5_6",
                    "--mathdial-finetuned", "./evaluations/mathdial_finetuned.json",
                    "--mathdial-base", f"{args.evaluation_dir}/mathdial_base.json",
                    "--mathdial-gpt4o", f"{args.evaluation_dir}/mathdial_gpt4o.json",
                    "--debugging-finetuned", f"{args.evaluation_dir}/debugging_finetuned.json",
                    "--debugging-base", f"{args.evaluation_dir}/debugging_base.json",
                    "--debugging-gpt4o", f"{args.evaluation_dir}/debugging_gpt4o.json",
                    "--tutorchat-finetuned", f"{args.evaluation_dir}/tutorchat_finetuned.json",
                    "--tutorchat-base", f"{args.evaluation_dir}/tutorchat_base.json",
                    "--tutorchat-gpt4o", f"{args.evaluation_dir}/tutorchat_gpt4o.json",
                    "--output-dir", f"{args.figures_dir}"])
