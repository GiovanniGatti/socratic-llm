import argparse
import json
import pathlib
from typing import Union

from bokeh.io import show
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GENERATE-FIGURES")
    parser.add_argument("--finetuned", type=pathlib.Path, help="Path to the evaluation of the fine-tuned model")
    parser.add_argument("--base", type=pathlib.Path, help="Path to the evaluation of the base model")
    parser.add_argument("--gpt4o", type=pathlib.Path, help="Path to the evaluation of the GPT-4o")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Path to directory where to store generated images")

    args = parser.parse_args()

    with open(args.finetuned) as f:
        finetuned = json.load(f)

    with open(args.base) as f:
        base = json.load(f)

    with open(args.gpt4o) as f:
        gpt4o = json.load(f)

    # Figure 4
    datasets = ['MathDial', ]
    model_name = ['Base', 'Fine-tuned', 'GPT-4o']

    base_score = round(sum(element["score"] for element in base) / len(base), 3)
    finetuned_score = round(sum(element["score"] for element in finetuned) / len(finetuned), 3)
    gpt4o_score = round(sum(element["score"] for element in gpt4o) / len(finetuned), 3)

    data = {
        'datasets': datasets,
        'base': [base_score, ],
        'finetuned': [finetuned_score, ],
        'gpt-4o': [gpt4o_score, ],
    }

    source = ColumnDataSource(data=data)

    p = figure(x_range=datasets, y_range=(.0, 1.2), toolbar_location=None, tools="", height=350)

    p.vbar(x=dodge('datasets', -0.25, range=p.x_range), top='finetuned', source=source,
           width=0.2, color="#718dbf", legend_label="Finetuned")
    p.vbar(x=dodge('datasets', 0, range=p.x_range), top='base', source=source,
           width=0.2, color="#c9d9d3", legend_label="Base")
    p.vbar(x=dodge('datasets', 0.25, range=p.x_range), top='gpt-4o', source=source,
           width=0.2, color="#e84d60", legend_label="GPT-4o")

    labels = LabelSet(x=dodge('datasets', -0.25, range=p.x_range), y='finetuned', text='finetuned', level='glyph',
                      text_align='center', y_offset=5, source=source)
    p.add_layout(labels)

    labels = LabelSet(x=dodge('datasets', 0., range=p.x_range), y='base', text='base', level='glyph',
                      text_align='center', y_offset=5, source=source)
    p.add_layout(labels)

    labels = LabelSet(x=dodge('datasets', 0.25, range=p.x_range), y='gpt-4o', text='gpt-4o', level='glyph',
                      text_align='center', y_offset=5, source=source)
    p.add_layout(labels)

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"

    show(p)

    # Figure 5
    datasets = ["question?", "on topic?", "helpful?", "reveal answer?"]
    model_name = ['Base', 'Fine-tuned', 'GPT-4o']

    base_scores = []
    finetuned_scores = []
    gpt4o_scores = []
    for metric in ("questions", "on_topic", "helpful", "reveal_answer"):
        def to_int(val: Union[str, int]) -> int:
            if isinstance(val, int):
                return val
            return "Yes" in val


        base_score = round(sum(to_int(element["evaluation"][metric]) for element in base) / len(base), 2)
        finetuned_score = round(sum(to_int(element["evaluation"][metric]) for element in finetuned) / len(finetuned), 2)
        gpt4o_score = round(sum(to_int(element["evaluation"][metric]) for element in gpt4o) / len(finetuned), 2)

        base_scores.append(base_score)
        finetuned_scores.append(finetuned_score)
        gpt4o_scores.append(gpt4o_score)

    data = {
        'datasets': datasets,
        'base': base_scores,
        'finetuned': finetuned_scores,
        'gpt-4o': gpt4o_scores,
    }

    source = ColumnDataSource(data=data)

    p = figure(x_range=datasets, toolbar_location=None, tools="")

    p.vbar(x=dodge('datasets', -0.25, range=p.x_range), top='finetuned', source=source,
           width=0.2, color="#718dbf", legend_label="Finetuned")
    p.vbar(x=dodge('datasets', 0, range=p.x_range), top='base', source=source,
           width=0.2, color="#c9d9d3", legend_label="Base")
    p.vbar(x=dodge('datasets', 0.25, range=p.x_range), top='gpt-4o', source=source,
           width=0.2, color="#e84d60", legend_label="GPT-4o")

    labels = LabelSet(x=dodge('datasets', -0.25, range=p.x_range), y='finetuned', text='finetuned', level='glyph',
                      text_align='center', y_offset=5, source=source)
    p.add_layout(labels)

    labels = LabelSet(x=dodge('datasets', 0., range=p.x_range), y='base', text='base', level='glyph',
                      text_align='center', y_offset=5, source=source)
    p.add_layout(labels)

    labels = LabelSet(x=dodge('datasets', 0.25, range=p.x_range), y='gpt-4o', text='gpt-4o', level='glyph',
                      text_align='center', y_offset=5, source=source)
    p.add_layout(labels)

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "vertical"

    show(p)
