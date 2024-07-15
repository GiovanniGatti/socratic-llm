import argparse
import os
import pathlib
import random

import scipy
from bokeh.io import export_svg, curdoc
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge
from pydantic import BaseModel

from data import Scores, Evaluation, Example, CrossValidationDataset
from figures.colors import HUMAN, GPT4o
from figures.theme import PROJECT_THEME


class CrossValidation(BaseModel):
    prompt: str
    output: str
    human: Evaluation
    gpt4o: Evaluation


def main(cross_validation_dataset_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    with open(cross_validation_dataset_path) as f:
        cross_validations = CrossValidationDataset.model_validate_json(f.read())

    curdoc().theme = PROJECT_THEME

    # FIGURE 2
    _copy = list(cross_validations.root)
    random.shuffle(_copy)
    _copy = _copy[:30]
    idx = [str(i) for i in range(len(_copy))]

    data = {
        "idx": idx,
        "human": [example.human.summary_score() for example in _copy],
        "gpt-4o": [example.gpt4o.summary_score() for example in _copy],
    }

    source = ColumnDataSource(data=data)

    fig2 = figure(x_range=idx, y_range=(0., 1.05), title="Summary scores for 30 samples",
                  height=350, toolbar_location=None, tools="")
    fig2.output_backend = "svg"

    fig2.vbar(x=dodge("idx", -0.06, range=fig2.x_range), top="human", source=source,
              width=0.05, color=HUMAN, legend_label="Human")

    fig2.vbar(x=dodge("idx", 0.06, range=fig2.x_range), top="gpt-4o", source=source,
              width=0.05, color=GPT4o, legend_label="GPT-4o")

    fig2.x_range.range_padding = 0.1
    fig2.xgrid.grid_line_color = None
    fig2.legend.location = "bottom_right"
    fig2.legend.orientation = "horizontal"

    filename = f"{output_dir}/fig2.svg"
    export_svg(fig2, filename=filename)
    os.chmod(filename, 0o755)

    # FIGURE 3
    data = {
        "humans": [example.human.summary_score() for example in cross_validations.get_valid()],
        "gpt-4o": [example.gpt4o.summary_score() for example in cross_validations.get_valid()],
    }

    pearson, _ = scipy.stats.pearsonr(
        x=[example.human.summary_score() for example in cross_validations.get_valid()],
        y=[example.gpt4o.summary_score() for example in cross_validations.get_valid()]
    )

    source = ColumnDataSource(data)

    fig3 = figure(width=600, height=600, y_range=(-0.05, 1.05), x_range=(-0.05, 1.05),
                  title=f"Pearson correlation={pearson:.2f}",
                  x_axis_label="GPT-4o Score", y_axis_label="Human score")
    fig3.output_backend = "svg"

    fig3.scatter(x="gpt-4o", y="humans", size=8, alpha=0.8, source=source)

    filename = f"{output_dir}/fig3.svg"
    export_svg(fig3, filename=filename)
    os.chmod(filename, 0o755)

    # FIGURE 4
    human_scores = Scores(
        root=[
            Example(prompt=e.prompt, output=e.output, raw_evaluation="", evaluation_error=None, evaluation=e.human)
            for e in cross_validations.get_valid()
        ]
    )
    gpt4o_scores = Scores(
        root=[
            Example(prompt=e.prompt, output=e.output, raw_evaluation="", evaluation_error=None, evaluation=e.gpt4o)
            for e in cross_validations.get_valid()
        ]
    )
    score_components = ["question?", "on topic?", "helpful?", "reveals answer?"]

    data = {
        "score_components": score_components,
        "humans": [human_scores.avg_questions(), human_scores.avg_on_topic(),
                   human_scores.avg_helpfulness(), human_scores.avg_reveal_answer()],
        "gpt-4o": [gpt4o_scores.avg_questions(), gpt4o_scores.avg_on_topic(),
                   gpt4o_scores.avg_helpfulness(), gpt4o_scores.avg_reveal_answer()],
    }

    source = ColumnDataSource(data=data)

    fig4 = figure(x_range=score_components, y_range=(0, 1.1), height=550, width=600, toolbar_location=None, tools="",
                  y_axis_label="Summary score")
    fig4.output_backend = "svg"

    fig4.vbar(x=dodge("score_components", -0.15, range=fig4.x_range), top="humans", source=source,
              width=0.2, color=HUMAN, legend_label="Humans")
    fig4.vbar(x=dodge("score_components", 0.15, range=fig4.x_range), top="gpt-4o", source=source,
              width=0.2, color=GPT4o, legend_label="GPT-4o")

    labels = LabelSet(x=dodge("score_components", -0.15, range=fig4.x_range), y="humans", text="humans",
                      level="glyph", text_align="center", y_offset=5, source=source)
    fig4.add_layout(labels)

    labels = LabelSet(x=dodge("score_components", 0.15, range=fig4.x_range), y="gpt-4o", text="gpt-4o",
                      level="glyph", text_align="center", y_offset=5, source=source)
    fig4.add_layout(labels)

    fig4.x_range.range_padding = 0.1
    fig4.xgrid.grid_line_color = None
    fig4.legend.location = "top_right"
    fig4.legend.orientation = "vertical"

    filename = f"{output_dir}/fig4.svg"
    export_svg(fig4, filename=filename)
    os.chmod(filename, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GENERATE-FIG-SEC5.1")
    parser.add_argument(
        "--cross-validation-dataset-path",
        required=True,
        type=pathlib.Path,
        help="Path to cross validation between humans and GPT"
    )
    parser.add_argument("--output-dir", type=pathlib.Path, help="Path to directory where to store generated images")

    args = parser.parse_args()

    main(args.cross_validatation_dataset_path, args.output_dir)
