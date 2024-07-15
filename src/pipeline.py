import argparse
import pathlib
import shutil
from pathlib import Path

import eval_model
import gen_train_dataset
import human_vs_gpt
import self_eval
import train
from figures import fig2_4, fig5_6, table
from tools import JudgeLLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PIPELINE",
        description="Runs training pipeline end-to-end"
    )
    parser.add_argument("--instruct-model", default="microsoft/Phi-3-mini-4k-instruct", type=str,
                        help="HF name of instruct model to finetune")
    parser.add_argument("--output-dir", required=True, type=str, help="Path where to store pipeline outputs")
    parser.add_argument(
        "--judge-llm", required=True, nargs=3, action=JudgeLLM,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct)."
    )
    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    args = parser.parse_args()

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

    INFERENCE_PROMPT = pathlib.Path("./templates/inference.txt")
    EVAL_PROMPT = pathlib.Path("./templates/judge_llm.txt")

    print(" ====== Generating DPO training datasets ====== ")
    for dataset in ("mathdial", "tutorchat"):
        target_dir = dpo_dir / dataset
        target_dir.mkdir(exist_ok=True)
        if not (target_dir / "train_dataset.json").exists():
            gen_train_dataset.main(dataset=pathlib.Path(f"./datasets/{dataset}_train.json"),
                                   inference_prompt=INFERENCE_PROMPT,
                                   eval_prompt=EVAL_PROMPT,
                                   instruct_model=args.instruct_model,
                                   judge_llm=args.judge_llm,
                                   output_path=pathlib.Path(f"{target_dir}/train_dataset.json"))

    print(" ====== Finetuning model with DPO ====== ")
    for dataset in ("mathdial", "tutorchat"):
        target_dir = dpo_dir / dataset
        target_dir.mkdir(exist_ok=True)

        checkpoint_dir = target_dir / "checkpoints"
        model_dir = target_dir / "model"

        if not model_dir.exists() or not any(model_dir.iterdir()):
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            model_dir.mkdir(exist_ok=True)
            checkpoint_dir.mkdir()

            train.main(dataset=pathlib.Path(f"{target_dir}/train_dataset.json"),
                       inference_prompt=INFERENCE_PROMPT,
                       instruct_model=args.instruct_model,
                       checkpoints_dir=checkpoint_dir,
                       model_dir=model_dir)

    print(" ====== Accessing model quality ====== ")
    for dataset in ("mathdial", "debugging", "tutorchat"):
        target_dir = evaluation_dir / dataset
        target_dir.mkdir(exist_ok=True)

        dataset_path = pathlib.Path(f"./datasets/{dataset}_test.json")
        from_finetuned_with_tutorchat = target_dir / "from_finetuned_with_tutorchat.json"
        from_finetuned_with_mathdial = target_dir / "from_finetuned_with_mathdial.json"
        base = target_dir / "base.json"
        gpt4o = target_dir / "gpt4o.json"

        if not from_finetuned_with_tutorchat.exists():
            eval_model.main(dataset=dataset_path,
                            inference_prompt=INFERENCE_PROMPT,
                            eval_prompt=EVAL_PROMPT,
                            model_path=dpo_dir / "tutorchat" / "model",
                            judge_llm=args.judge_llm,
                            output_path=from_finetuned_with_tutorchat)

        if not from_finetuned_with_mathdial.exists():
            eval_model.main(dataset=dataset_path,
                            inference_prompt=INFERENCE_PROMPT,
                            eval_prompt=EVAL_PROMPT,
                            model_path=dpo_dir / "mathdial" / "model",
                            judge_llm=args.judge_llm,
                            output_path=from_finetuned_with_mathdial)

        if not base.exists():
            eval_model.main(dataset=dataset_path,
                            inference_prompt=INFERENCE_PROMPT,
                            eval_prompt=EVAL_PROMPT,
                            model_path=args.instruct_model,
                            judge_llm=args.judge_llm,
                            output_path=base)

        if not gpt4o.exists():
            self_eval.main(dataset=dataset_path,
                           inference_prompt=INFERENCE_PROMPT,
                           eval_prompt=EVAL_PROMPT,
                           judge_llm=args.judge_llm,
                           output_path=gpt4o)

    print(" ====== Human vs. GPT analysis ====== ")
    human_vs_gpt_path = evaluation_dir / "human_vs_gpt.json"

    if not human_vs_gpt_path.exists():
        human_vs_gpt.main(humans=pathlib.Path("./datasets/mathdial_human_eval.json"),
                          eval_prompt=EVAL_PROMPT,
                          judge_llm=args.judge_llm,
                          output_path=human_vs_gpt_path)

    print(" ====== Generating analysis ====== ")
    fig2_4.main(cross_validation_dataset_path=human_vs_gpt_path,
                output_dir=figures_dir)

    fig5_6.main(mathdial_finetuned=evaluation_dir / "mathdial" / "from_finetuned_with_tutorchat.json",
                mathdial_base=evaluation_dir / "mathdial" / "base.json",
                mathdial_gpt4o=evaluation_dir / "mathdial" / "gpt4o.json",
                debugging_finetuned=evaluation_dir / "debugging" / "from_finetuned_with_tutorchat.json",
                debugging_base=evaluation_dir / "debugging" / "base.json",
                debugging_gpt4o=evaluation_dir / "debugging" / "gpt4o.json",
                tutorchat_finetuned=evaluation_dir / "tutorchat" / "from_finetuned_with_tutorchat.json",
                tutorchat_base=evaluation_dir / "tutorchat" / "base.json",
                tutorchat_gpt4o=evaluation_dir / "tutorchat" / "gpt4o.json",
                output_dir=figures_dir)

    table.main(mathdial_finetuned_with_tutorchat=evaluation_dir / "mathdial" / "from_finetuned_with_tutorchat.json",
               mathdial_finetuned_with_mathdial=evaluation_dir / "mathdial" / "from_finetuned_with_mathdial.json",
               debugging_finetuned_with_tutorchat=evaluation_dir / "debugging" / "from_finetuned_with_tutorchat.json",
               debugging_finetuned_with_mathdial=evaluation_dir / "debugging" / "from_finetuned_with_mathdial.json",
               tutorchat_finetuned_with_tutorchat=evaluation_dir / "tutorchat" / "from_finetuned_with_tutorchat.json",
               tutorchat_finetuned_with_mathdial=evaluation_dir / "tutorchat" / "from_finetuned_with_mathdial.json",
               output_dir=figures_dir)
