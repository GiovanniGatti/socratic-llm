import argparse
import os
import pathlib

from bokeh.io import export_svg
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge

from data import Scores
from figures.colors import FINETUNED, BASE, GPT4o

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GENERATE-FIGURES")
    parser.add_argument("--finetuned", type=pathlib.Path, help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--base", type=pathlib.Path, help="Path to the evaluation of the base model")
    parser.add_argument("--gpt4o", type=pathlib.Path, help="Path to the evaluation of the GPT-4o")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Path to directory where to store generated images")

    args = parser.parse_args()

    with open(args.finetuned, "r") as f:
        finetuned = Scores.model_validate_json(f.read())

    with open(args.base, "r") as f:
        base = Scores.model_validate_json(f.read())

    with open(args.gpt4o, "r") as f:
        gpt4o = Scores.model_validate_json(f.read())

    # Figure 5
    datasets = ["MathDial", ]
    model_name = ["Base", "Fine-tuned", "GPT-4o"]

    data = {
        "datasets": datasets,
        "base": [round(base.avg_summary_score(), 2), ],
        "finetuned": [round(finetuned.avg_summary_score(), 2), ],
        "gpt-4o": [round(gpt4o.avg_summary_score(), 2), ],
    }

    source = ColumnDataSource(data=data)

    fig5 = figure(x_range=datasets, y_range=(.0, 1.1), toolbar_location=None, tools="", height=450)

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

    filename = f"{args.output_dir}/fig5.svg"
    export_svg(fig5, filename=filename)
    os.chmod(filename, 0o755)

    # Figure 6
    datasets = ["question?", "on topic?", "helpful?", "reveal answer?"]
    model_name = ["Base", "Fine-tuned", "GPT-4o"]

    data = {
        "datasets": datasets,
        "base": [round(base.avg_questions(), 2), round(base.avg_on_topic() / 5, 2),
                 round(base.avg_helpfulness() / 5, 2), round(base.avg_reveal_answer(), 2)],
        "finetuned": [round(finetuned.avg_questions(), 2), round(finetuned.avg_on_topic() / 5, 2),
                      round(finetuned.avg_helpfulness() / 5, 2), round(finetuned.avg_reveal_answer(), 2)],
        "gpt-4o": [round(gpt4o.avg_questions(), 2), round(gpt4o.avg_on_topic() / 5, 2),
                   round(gpt4o.avg_helpfulness() / 5, 2), round(gpt4o.avg_reveal_answer(), 2)],
    }

    source = ColumnDataSource(data=data)

    fig6 = figure(x_range=datasets, y_range=(0, 1.1), height=550, width=600, toolbar_location=None, tools="")
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

    filename = f"{args.output_dir}/fig6.svg"
    export_svg(fig6, filename=filename)
    os.chmod(filename, 0o755)
