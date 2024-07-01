import argparse
import pathlib
from typing import List

import scipy
from bokeh.io import show, export_svg
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from openai import OpenAI
from tqdm import tqdm

from data import Scores, Evaluation
from tools import escape_template


class CrossValidation:

    def __init__(self, prompt: str, output: str, human: Evaluation, gpt4o: Evaluation):
        self.prompt = prompt
        self.output = output
        self.human = human
        self.gpt4o = gpt4o


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GENERATE-FIG-HUMANS-VS-GPT4o")
    parser.add_argument("--humans", type=pathlib.Path, help="Path to the evaluation by humans")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument("--openai-api-key", required=True, type=str, help="Open AI api key")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Path to directory where to store generated images")

    args = parser.parse_args()

    with open(args.eval_prompt, 'r', encoding='utf-8') as file:
        judge_llm_prompt = escape_template(file.read())

    with open(args.humans) as f:
        human_eval = Scores.model_validate_json(f.read())

    client = OpenAI(api_key=args.openai_api_key)
    cross_validations: List[CrossValidation] = []
    for example in tqdm(human_eval[:3]):
        answer = example.output
        student = answer.split("Student")[0] if "Student" in answer else answer
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": judge_llm_prompt.format(conversation=example.prompt, answer=student)},
            ],
            model="gpt-4o",
            temperature=0.2,
            seed=0,
        )
        evaluation = Evaluation.model_validate_json(chat_completion.choices[0].message.content)
        cross_validation = CrossValidation(example.prompt, example.output, human=example.evaluation, gpt4o=evaluation)
        cross_validations.append(cross_validation)

    data = {
        "humans": [example.human.get_score() for example in cross_validations],
        "gpt-4o": [example.gpt4o.get_score() for example in cross_validations],
    }

    pearson, _ = scipy.stats.pearsonr(
        x=[example.human.get_score() for example in cross_validations],
        y=[example.gpt4o.get_score() for example in cross_validations]
    )

    source = ColumnDataSource(data)

    plot = figure(width=600, height=600, y_range=(-0.05, 1.05), x_range=(-0.05, 1.05),
                  title=f"Pearson correlation={pearson:.2f}",
                  x_axis_label="GPT-4o Score", y_axis_label="Human score")
    plot.output_backend = "svg"

    plot.scatter(x="gpt-4o", y="humans", size=8, alpha=0.8, source=source)

    export_svg(plot, filename=f"{args.output_dir}/fig3.svg")
