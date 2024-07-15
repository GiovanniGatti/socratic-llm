import argparse
import os
import pathlib

from bokeh.io import export_svg, curdoc
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge

from data import Scores
from figures.colors import FINETUNED, BASE, GPT4o
from figures.theme import PROJECT_THEME


class EnsembleDataset:
    def __init__(self):
        self.mathdial = dict()
        self.debugging = dict()
        self.tutorchat = dict()


def main(
        mathdial_finetuned: pathlib.Path,
        mathdial_base: pathlib.Path,
        mathdial_gpt4o: pathlib.Path,
        debugging_finetuned: pathlib.Path,
        debugging_base: pathlib.Path,
        debugging_gpt4o: pathlib.Path,
        tutorchat_finetuned: pathlib.Path,
        tutorchat_base: pathlib.Path,
        tutorchat_gpt4o: pathlib.Path,
        output_dir: pathlib.Path
) -> None:
    all_paths = locals()
    ensemble_dataset = EnsembleDataset()
    for dataset in ("mathdial", "debugging", "tutorchat"):
        for model in ("finetuned", "base", "gpt4o"):
            with open(all_paths[f"{dataset}_{model}"], "r") as f:
                scores = Scores.model_validate_json(f.read())
            getattr(ensemble_dataset, dataset)[model] = scores

    curdoc().theme = PROJECT_THEME

    # Figure 5
    datasets = ["MathDial", "Debugging", "TutorChat"]
    model_name = ["Base", "Fine-tuned", "GPT-4o"]

    data = {
        "datasets": datasets,
        "base": [getattr(ensemble_dataset, dataset)["base"].avg_summary_score() for dataset in
                 ("mathdial", "debugging", "tutorchat")],
        "finetuned": [getattr(ensemble_dataset, dataset)["finetuned"].avg_summary_score() for dataset in
                      ("mathdial", "debugging", "tutorchat")],
        "gpt-4o": [getattr(ensemble_dataset, dataset)["gpt4o"].avg_summary_score() for dataset in
                   ("mathdial", "debugging", "tutorchat")],
    }

    source = ColumnDataSource(data=data)

    fig5 = figure(height=550, width=600, x_range=datasets, y_range=(.0, 1.1), toolbar_location=None, tools="",
                  y_axis_label="Summary score")
    fig5.output_backend = "svg"

    fig5.vbar(x=dodge("datasets", -0.25, range=fig5.x_range), top="finetuned", source=source,
              width=0.2, color=FINETUNED, legend_label="Finetuned")
    fig5.vbar(x=dodge("datasets", 0, range=fig5.x_range), top="base", source=source,
              width=0.2, color=BASE, legend_label="Base")
    fig5.vbar(x=dodge("datasets", 0.25, range=fig5.x_range), top="gpt-4o", source=source,
              width=0.2, color=GPT4o, legend_label="GPT-4o")

    labels = LabelSet(x=dodge("datasets", -0.25, range=fig5.x_range), y="finetuned", text="finetuned", level="glyph",
                      text_align="center", y_offset=5, source=source)
    fig5.add_layout(labels)

    labels = LabelSet(x=dodge("datasets", 0., range=fig5.x_range), y="base", text="base", level="glyph",
                      text_align="center", y_offset=5, source=source)
    fig5.add_layout(labels)

    labels = LabelSet(x=dodge("datasets", 0.25, range=fig5.x_range), y="gpt-4o", text="gpt-4o", level="glyph",
                      text_align="center", y_offset=5, source=source)
    fig5.add_layout(labels)

    fig5.x_range.range_padding = 0.1
    fig5.xgrid.grid_line_color = None
    fig5.legend.location = "bottom_right"
    fig5.legend.orientation = "horizontal"

    filename = f"{output_dir}/fig5.svg"
    export_svg(fig5, filename=filename)
    os.chmod(filename, 0o755)

    # Figure 6
    datasets = ["question?", "on topic?", "helpful?", "reveal answer?"]
    model_name = ["Base", "Fine-tuned", "GPT-4o"]

    data = {
        "datasets": datasets,
        "base": [getattr(ensemble_dataset.mathdial["base"], metric)() for metric in
                 ("avg_questions", "avg_on_topic", "avg_helpfulness", "avg_reveal_answer")],
        "finetuned": [getattr(ensemble_dataset.mathdial["finetuned"], metric)() for metric in
                      ("avg_questions", "avg_on_topic", "avg_helpfulness", "avg_reveal_answer")],
        "gpt-4o": [getattr(ensemble_dataset.mathdial["gpt4o"], metric)() for metric in
                   ("avg_questions", "avg_on_topic", "avg_helpfulness", "avg_reveal_answer")],
    }

    source = ColumnDataSource(data=data)

    fig6 = figure(height=550, width=600, x_range=datasets, y_range=(0, 1.1), toolbar_location=None, tools="",
                  y_axis_label="Normalized score")
    fig6.output_backend = "svg"

    fig6.vbar(x=dodge("datasets", -0.25, range=fig6.x_range), top="finetuned", source=source,
              width=0.2, color=FINETUNED, legend_label="Finetuned")
    fig6.vbar(x=dodge("datasets", 0, range=fig6.x_range), top="base", source=source,
              width=0.2, color=BASE, legend_label="Base")
    fig6.vbar(x=dodge("datasets", 0.25, range=fig6.x_range), top="gpt-4o", source=source,
              width=0.2, color=GPT4o, legend_label="GPT-4o")

    labels = LabelSet(x=dodge("datasets", -0.25, range=fig6.x_range), y="finetuned", text="finetuned", level="glyph",
                      text_align="center", y_offset=5, source=source)
    fig6.add_layout(labels)

    labels = LabelSet(x=dodge("datasets", 0., range=fig6.x_range), y="base", text="base", level="glyph",
                      text_align="center", y_offset=5, source=source)
    fig6.add_layout(labels)

    labels = LabelSet(x=dodge("datasets", 0.25, range=fig6.x_range), y="gpt-4o", text="gpt-4o", level="glyph",
                      text_align="center", y_offset=5, source=source)
    fig6.add_layout(labels)

    fig6.x_range.range_padding = 0.1
    fig6.xgrid.grid_line_color = None
    fig6.legend.location = "top_right"
    fig6.legend.orientation = "vertical"

    filename = f"{output_dir}/fig6.svg"
    export_svg(fig6, filename=filename)
    os.chmod(filename, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="FIG-SEC6")
    parser.add_argument("--mathdial-finetuned", type=pathlib.Path,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--mathdial-base", type=pathlib.Path, help="Path to the evaluation of the base model")
    parser.add_argument("--mathdial-gpt4o", type=pathlib.Path, help="Path to the evaluation of the GPT-4o")

    parser.add_argument("--debugging-finetuned", type=pathlib.Path,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--debugging-base", type=pathlib.Path, help="Path to the evaluation of the base model")
    parser.add_argument("--debugging-gpt4o", type=pathlib.Path, help="Path to the evaluation of the GPT-4o")

    parser.add_argument("--tutorchat-finetuned", type=pathlib.Path,
                        help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--tutorchat-base", type=pathlib.Path, help="Path to the evaluation of the base model")
    parser.add_argument("--tutorchat-gpt4o", type=pathlib.Path, help="Path to the evaluation of the GPT-4o")

    parser.add_argument("--output-dir", type=pathlib.Path, help="Path to directory where to store generated images")

    args = parser.parse_args()

    main(args.mathdial_finetuned,
         args.mathdial_base,
         args.mathdial_gpt4o,
         args.debugging_finetuned,
         args.debugging_base,
         args.debugging_gpt4o,
         args.tutorchat_finetuned,
         args.tutorchat_base,
         args.tutorchat_gpt4o,
         args.output_dir)
