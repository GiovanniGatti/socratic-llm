import argparse
import os
import shutil
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="PIPELINE")
    parser.add_argument("--openai-api-key", required=False, type=str, help="Open AI api key")
    parser.add_argument("--output-dir", required=True, type=str, help="Path where to store pipeline outputs")
    parser.add_argument("--instruct-model", default="microsoft/Phi-3-mini-4k-instruct", type=str,
                        help="HF name of instruct model to finetune")
    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    args = parser.parse_args()

    OPENAI_API_KEY = args.openai_api_key
    if OPENAI_API_KEY is None:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    if OPENAI_API_KEY is None:
        raise ValueError("Must provide OPENAI_API_KEY either through command line or environment variable")

    output_dir = Path(args.output_dir)
    if not args.use_cache:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    dpo_dir = output_dir / "dpo"
    evaluation_dir = output_dir / "evaluation"
    figures_dir = output_dir / "figures"

    dpo_dir.mkdir(exist_ok=True)
    evaluation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Generate training data for DPO
    for dataset in ("mathdial", "tutorchat"):
        target_dir = dpo_dir / dataset
        target_dir.mkdir(exist_ok=True)
        if not (target_dir / "train_dataset.json").exists():
            subprocess.run(["python", "-m", "gen_train_dataset",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--instruct-model", args.instruct_model,
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{target_dir}/train_dataset.json"])

    # Perform DPO training
    for dataset in ("mathdial", "tutorchat"):
        target_dir = dpo_dir / dataset
        target_dir.mkdir(exist_ok=True)

        checkpoint_dir = target_dir / "checkpoints"
        model_dir = target_dir / "model"

        if not model_dir.exists():
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            model_dir.mkdir()
            checkpoint_dir.mkdir()

            subprocess.run(["python", "-m", "train",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--instruct-model", args.instruct_model,
                            "--checkpoints-dir", f"{checkpoint_dir}",
                            "--model-dir", f"{model_dir}"])

    # Assess model quality
    for dataset in ("mathdial", "debugging", "tutorchat"):
        target_dir = evaluation_dir / dataset
        target_dir.mkdir(exist_ok=True)

        from_finetuned_with_tutorchat = target_dir / "from_finetuned_with_tutorchat.json"
        from_finetuned_with_mathdial = target_dir / "from_finetuned_with_mathdial.json"
        base = target_dir / "base.json"
        gpt4o = target_dir / "gpt4o.json"

        if not from_finetuned_with_tutorchat.exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--model-path", f"{dpo_dir / 'tutorchat' / 'model'}",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{from_finetuned_with_tutorchat}"])

        if not from_finetuned_with_mathdial.exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--model-path", f"{dpo_dir / 'mathdial' / 'model'}",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{from_finetuned_with_mathdial}"])

        if not base.exists():
            subprocess.run(["python", "-m", "eval_model",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--peft-adapter", args.instruct_model,
                            "--output", f"{base}"])

        if not gpt4o.exists():
            subprocess.run(["python", "-m", "eval_gpt_4o",
                            "--input", f"./datasets/{dataset}.json",
                            "--inference-prompt", "./templates/inference.txt",
                            "--eval-prompt", "./templates/judge_llm.txt",
                            "--openai-api-key", OPENAI_API_KEY,
                            "--output", f"{gpt4o}"])

    # Analysis
    if not len(list(figures_dir.glob("fig[2-4].svg"))) > 0:
        subprocess.run(["python", "-m", "figures.fig2_4",
                        "--humans", f"./datasets/mathdial_human_eval.json",
                        "--eval-prompt", "./templates/judge_llm.txt",
                        "--openai-api-key", OPENAI_API_KEY,
                        "--output", f"{figures_dir}"])

    subprocess.run(["python", "-m", "figures.fig5_6",
                    "--mathdial-finetuned", f"{evaluation_dir}/mathdial/from_finetuned_with_tutorchat.json",
                    "--mathdial-base", f"{evaluation_dir}/mathdial/base.json",
                    "--mathdial-gpt4o", f"{evaluation_dir}/mathdial/gpt4o.json",
                    "--debugging-finetuned", f"{evaluation_dir}/debugging/from_finetuned_with_tutorchat.json",
                    "--debugging-base", f"{evaluation_dir}/debugging/base.json",
                    "--debugging-gpt4o", f"{evaluation_dir}/debugging/gpt4o.json",
                    "--tutorchat-finetuned", f"{evaluation_dir}/tutorchat/from_finetuned_with_tutorchat.json",
                    "--tutorchat-base", f"{evaluation_dir}/tutorchat/base.json",
                    "--tutorchat-gpt4o", f"{evaluation_dir}/tutorchat/gpt4o.json",
                    "--output-dir", f"{figures_dir}"])

    subprocess.run(["python", "-m", "figures.table",
                    "--mathdial-finetuned-with-tutorchat",
                    f"{evaluation_dir}/mathdial/from_finetuned_with_tutorchat.json",
                    "--debugging-finetuned-with-tutorchat",
                    f"{evaluation_dir}/debugging/from_finetuned_with_tutorchat.json",
                    "--tutorchat-finetuned-with-tutorchat",
                    f"{evaluation_dir}/tutorchat/from_finetuned_with_tutorchat.json",
                    "--mathdial-finetuned-with-mathdial",
                    f"{evaluation_dir}/mathdial/from_finetuned_with_mathdial.json",
                    "--debugging-finetuned-with-mathdial",
                    f"{evaluation_dir}/debugging/from_finetuned_with_mathdial.json",
                    "--tutorchat-finetuned-with-mathdial",
                    f"{evaluation_dir}/tutorchat/from_finetuned_with_mathdial.json",
                    "--output-dir", f"{figures_dir}"])
