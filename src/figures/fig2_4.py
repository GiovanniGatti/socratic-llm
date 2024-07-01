import argparse
import os
import pathlib
import random
from typing import List

import scipy
from bokeh.io import export_svg
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge
from openai import OpenAI
from tqdm import tqdm

from data import Scores, Evaluation, Example
from figures.colors import HUMAN, GPT4o
from tools import escape_template


class CrossValidation:

    def __init__(self, prompt: str, output: str, human: Evaluation, gpt4o: Evaluation):
        self.prompt = prompt
        self.output = output
        self.human = human
        self.gpt4o = gpt4o


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GENERATE-FIG-SEC5.1")
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
    for example in tqdm(human_eval):
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

    # FIGURE 2
    _copy = list(cross_validations)
    random.shuffle(_copy)
    _copy = _copy[:30]
    idx = [str(i) for i in range(len(_copy))]

    data = {
        'idx': idx,
        'human': [example.human.summary_score() for example in _copy],
        'gpt-4o': [example.gpt4o.summary_score() for example in _copy],
    }

    source = ColumnDataSource(data=data)

    fig2 = figure(x_range=idx, y_range=(0., 1.05), title="Summary scores for 30 samples",
                  height=350, toolbar_location=None, tools="")
    fig2.output_backend = "svg"

    fig2.vbar(x=dodge('idx', -0.06, range=fig2.x_range), top='human', source=source,
              width=0.05, color=HUMAN, legend_label="Human")

    fig2.vbar(x=dodge('idx', 0.06, range=fig2.x_range), top='gpt-4o', source=source,
              width=0.05, color=GPT4o, legend_label="GPT-4o")

    fig2.x_range.range_padding = 0.1
    fig2.xgrid.grid_line_color = None
    fig2.legend.location = "bottom_right"
    fig2.legend.orientation = "horizontal"

    filename = f"{args.output_dir}/fig2.svg"
    export_svg(fig2, filename=filename)
    os.chmod(filename, 0o755)

    # FIGURE 3
    data = {
        "humans": [example.human.summary_score() for example in cross_validations],
        "gpt-4o": [example.gpt4o.summary_score() for example in cross_validations],
    }

    pearson, _ = scipy.stats.pearsonr(
        x=[example.human.summary_score() for example in cross_validations],
        y=[example.gpt4o.summary_score() for example in cross_validations]
    )

    source = ColumnDataSource(data)

    fig3 = figure(width=600, height=600, y_range=(-0.05, 1.05), x_range=(-0.05, 1.05),
                  title=f"Pearson correlation={pearson:.2f}",
                  x_axis_label="GPT-4o Score", y_axis_label="Human score")
    fig3.output_backend = "svg"

    fig3.scatter(x="gpt-4o", y="humans", size=8, alpha=0.8, source=source)

    filename = f"{args.output_dir}/fig3.svg"
    export_svg(fig3, filename=filename)
    os.chmod(filename, 0o755)

    # FIGURE 4
    human_scores = Scores(
        root=[Example(prompt=e.prompt, output=e.output, evaluation=e.human) for e in cross_validations]
    )
    gpt4o_scores = Scores(
        root=[Example(prompt=e.prompt, output=e.output, evaluation=e.gpt4o) for e in cross_validations]
    )
    score_components = ["question?", "on topic?", "helpful?", "reveals answer?"]
    model_name = ['Human', 'GPT-4o']

    data = {
        'score_components': score_components,
        'humans': [round(human_scores.avg_questions(), 2), round(human_scores.avg_on_topic() / 5, 2),
                   round(human_scores.avg_helpfulness() / 5, 2), round(human_scores.avg_reveal_answer(), 2)],
        'gpt-4o': [round(gpt4o_scores.avg_questions(), 2), round(gpt4o_scores.avg_on_topic() / 5, 2),
                   round(gpt4o_scores.avg_helpfulness() / 5, 2), round(gpt4o_scores.avg_reveal_answer(), 2)],
    }

    source = ColumnDataSource(data=data)

    fig4 = figure(x_range=score_components, y_range=(0, 1.1), height=550, width=600, toolbar_location=None, tools="")
    fig4.output_backend = "svg"

    fig4.vbar(x=dodge('score_components', -0.15, range=fig4.x_range), top='humans', source=source,
              width=0.2, color=HUMAN, legend_label="Humans")
    fig4.vbar(x=dodge('score_components', 0.15, range=fig4.x_range), top='gpt-4o', source=source,
              width=0.2, color=GPT4o, legend_label="GPT-4o")

    labels = LabelSet(x=dodge('score_components', -0.15, range=fig4.x_range), y='humans', text='humans',
                      level='glyph', text_align='center', y_offset=5, source=source)
    fig4.add_layout(labels)

    labels = LabelSet(x=dodge('score_components', 0.15, range=fig4.x_range), y='gpt-4o', text='gpt-4o',
                      level='glyph', text_align='center', y_offset=5, source=source)
    fig4.add_layout(labels)

    fig4.x_range.range_padding = 0.1
    fig4.xgrid.grid_line_color = None
    fig4.legend.location = "top_right"
    fig4.legend.orientation = "vertical"

    filename = f"{args.output_dir}/fig4.svg"
    export_svg(fig4, filename=filename)
    os.chmod(filename, 0o755)
