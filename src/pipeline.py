import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="PIPELINE")
    parser.add_argument("--openai-api-key", required=False, type=str, help="Open AI api key")
    parser.add_argument("--evaluation-dir", required=True, type=str, help="Path where to store assessments")
    parser.add_argument("--figures-dir", required=True, type=str, help="Path where to store figures")
    args = parser.parse_args()

    OPENAI_API_KEY = args.openai_api_key
    if OPENAI_API_KEY is None:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    if OPENAI_API_KEY is None:
        raise ValueError("Must provide OPENAI_API_KEY either through command line or environment variable")

    for dataset in ("mathdial", "debugging", "tutorchat"):
        subprocess.run(["python", "-m", "eval_model",
                        "--input", f"./datasets/{dataset}.json",
                        "--inference-prompt", "./templates/inference.txt",
                        "--eval-prompt", "./templates/judge_llm.txt",
                        "--openai-api-key", OPENAI_API_KEY,
                        "--output", f"{args.evaluation_dir}/{dataset}_base.json"])

        subprocess.run(["python", "-m", "eval_model",
                        "--input", f"./datasets/{dataset}.json",
                        "--inference-prompt", "./templates/inference.txt",
                        "--eval-prompt", "./templates/judge_llm.txt",
                        "--openai-api-key", OPENAI_API_KEY,
                        "--without-lora-adapter",
                        "--output", f"{args.evaluation_dir}/{dataset}_finetuned.json"])

        subprocess.run(["python", "-m", "eval_gpt_4o",
                        "--input", f"./datasets/{dataset}.json",
                        "--inference-prompt", "./templates/inference.txt",
                        "--eval-prompt", "./templates/judge_llm.txt",
                        "--openai-api-key", OPENAI_API_KEY,
                        "--output", f"{args.evaluation_dir}/{dataset}_gpt4o.json"])

    subprocess.run(["python", "-m", "figures.fig2_4",
                    "--humans", f"./datasets/mathdial_human_eval.json",
                    "--eval-prompt", "./templates/judge_llm.txt",
                    "--openai-api-key", OPENAI_API_KEY,
                    "--output", f"{args.figures_dir}"])

    subprocess.run(["python", "-m", "figures.fig5_6",
                    "--finetuned", f"{args.evaluation_dir}/mathdial_finetuned.json",
                    "--base", f"{args.evaluation_dir}/mathdial_base.json",
                    "--gpt4o", f"{args.evaluation_dir}/mathdial_gpt4o.json",
                    "--output", f"{args.figures_dir}"])
