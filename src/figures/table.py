import argparse
import os
import pathlib
from collections import defaultdict

import pandas as pd

from data import Scores


def main(
        mathdial_finetuned_with_tutorchat: pathlib.Path,
        mathdial_finetuned_with_mathdial: pathlib.Path,
        mathdial_finetuned_with_debugging: pathlib.Path,
        debugging_finetuned_with_tutorchat: pathlib.Path,
        debugging_finetuned_with_mathdial: pathlib.Path,
        debugging_finetuned_with_debugging: pathlib.Path,
        tutorchat_finetuned_with_tutorchat: pathlib.Path,
        tutorchat_finetuned_with_mathdial: pathlib.Path,
        tutorchat_finetuned_with_debugging: pathlib.Path,
        output_dir: pathlib.Path
) -> None:
    all_paths = locals()
    ensemble_dataset = defaultdict(list)
    for model in ("finetuned_with_tutorchat", "finetuned_with_mathdial", "finetuned_with_debugging"):
        for dataset in ("mathdial", "debugging", "tutorchat"):
            with open(all_paths[f"{dataset}_{model}"], "r") as f:
                scores = Scores.model_validate_json(f.read())
            ensemble_dataset[model].append(scores.avg_summary_score())

    df = pd.DataFrame(ensemble_dataset, index=["mathdial", "debugging", "tutorchat"])

    filename = f"{output_dir}/table.json"
    with open(filename, "w") as f:
        f.write(df.to_json(indent=2))
    os.chmod(filename, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TABLE-SEC6")
    parser.add_argument("--mathdial-finetuned-with-tutorchat", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--mathdial-finetuned-with-mathdial", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--mathdial-finetuned-with-debugging", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")

    parser.add_argument("--debugging-finetuned-with-tutorchat", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--debugging-finetuned-with-mathdial", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--debugging-finetuned-with-debugging", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")

    parser.add_argument("--tutorchat-finetuned-with-tutorchat", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--tutorchat-finetuned-with-mathdial", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--tutorchat-finetuned-with-debugging", type=pathlib.Path, required=True,
                        help="Path to the evaluation of the fine-tuned model")

    parser.add_argument("--output-dir", type=pathlib.Path, help="Path to directory where to store generated table")

    args = parser.parse_args()

    main(args.mathdial_finetuned_with_tutorchat,
         args.mathdial_finetuned_with_mathdial,
         args.mathdial_finetuned_with_debugging,
         args.debugging_finetuned_with_tutorchat,
         args.debugging_finetuned_with_mathdial,
         args.debugging_finetuned_with_debugging,
         args.tutorchat_finetuned_with_tutorchat,
         args.tutorchat_finetuned_with_mathdial,
         args.tutorchat_finetuned_with_debugging,
         args.output_dir)
