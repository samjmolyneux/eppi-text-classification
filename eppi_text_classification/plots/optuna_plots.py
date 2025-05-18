import re
from typing import NamedTuple

import numpy as np
import optuna
import plotly
from optuna.visualization._rank import _get_rank_info
from scipy.stats import rankdata

standard_config = {
    "scrollZoom": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "zoom",
        "pan",
        "select",
        "lasso2d",
        "autoscale",
        "zoom2d",
        "zoomIn2d",
        "zoomOut2d",
        "resetScale2d",
    ],
    "modeBarButtonsToAdd": ["toImage"],
    "displaylogo": False,
}


def make_tick_labels_4f(tick_labels):
    pattern = re.compile(r"\(([-+]?[0-9]*\.?[0-9]+)\)")

    formatted = []
    for label in tick_labels:
        match = pattern.search(label)
        if match:
            num_str = match.group(1)
            formatted_num = f"{float(num_str):.4f}"
            new_s = pattern.sub(f"({formatted_num})", label)
            formatted.append(new_s)
        else:
            formatted.append(label)

    return formatted


def create_opt_history_html(study, savepath):
    fig = optuna.visualization.plot_optimization_history(study)

    fig.update_layout(dragmode=False)

    fig.update_layout(
        title={
            "text": "Optimisation History Plot",
            "x": 0.5,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "center",
        },
    )

    fig.write_html(file=savepath, config=standard_config)


def create_param_importance_html(study, savepath):
    fig = optuna.visualization.plot_param_importances(study)

    fig.update_layout(dragmode=False)

    fig.update_layout(
        title={
            "text": "Parameter Importance Plot",
            "x": 0.5,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "center",
        },
    )

    fig.write_html(file=savepath, config=standard_config)


def create_timeline_html(study, savepath):
    fig = optuna.visualization.plot_timeline(study)

    fig.update_layout(hovermode=False, dragmode=False)

    fig.update_layout(
        title={
            "text": "Timeline Plot",
            "x": 0.5,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "center",
        },
    )

    fig.write_html(file=savepath, config=standard_config)


def create_all_optuna_plots(study, save_dir):
    create_opt_history_html(study, f"{save_dir}/opt_history.html")
    create_param_importance_html(study, f"{save_dir}/param_importance.html")
    create_timeline_html(study, f"{save_dir}/timeline.html")
    create_slice_plots_html(study, f"{save_dir}")


def create_slice_plots_html(study, savepath, yaxis_title="ROC_AUC"):
    # THIS METHOD ASSUMES THAT STUDY.GET_TRIALS() RETURNS TRIALS IN ORDER OF TRIAL NUMBER
    # AND THAT SLICE PLOT PLOTS THEM IN ORDER OF TRIAL NUMBER.
    # IF THIS METHOD EVERY BREAKS, THIS IS LIKELY THE REASON.

    param_names = set()
    for trial in study.trials:
        param_names.update(trial.params.keys())

    # Go through all trials, with the trial number.
    # Order them by the objective value
    # Map each trial to a color.
    # Draw the slice plot for a a param.
    # Then add the colors and color bar.

    trials = study.get_trials()
    objective_values = np.array([trial.value for trial in trials])

    # Get colour by rank for each trial and in the correct order
    colormap = "RdYlBu_r"
    colors = np.array(
        plotly.colors.sample_colorscale(
            colormap,
            len(trials),
        )
    )
    trial_ranks = rankdata(objective_values, method="average").astype(int) - 1
    colors = colors[trial_ranks]

    colourbar_tick_info = get_tick_info(objective_values)
    colourbar_tick_labels = make_tick_labels_4f(colourbar_tick_info.text)

    for param in param_names:
        fig = optuna.visualization.plot_slice(study, params=[param])

        # We get the colour for each trial
        # This approach assumes that the optuna.visualization.plot_slice method
        # Plots the data in order of trial number
        # But I could probably make it more robuts by getting the y's from the fig then mapping the
        # y's to the rank and then using that to get the color!!!!!!!!!
        params_trial_numbers = np.array(
            [trial.number for trial in trials if param in trial.params]
        )
        params_trial_colors = colors[params_trial_numbers]

        fig.update_traces(
            marker={
                "color": params_trial_colors,
                "line": {"width": 0.5, "color": "Grey"},
                "colorbar": {
                    "thickness": 25,
                    "tickvals": colourbar_tick_info.coloridxs,
                    "ticktext": colourbar_tick_labels,
                    "tickfont": {"size": 10},
                    "title": "",
                    "xpad": 0,
                    "x": 1.06,
                    "xref": "paper",
                },
                "colorscale": "RdYlBu_r",
                "cmin": 0,
                "cmax": 1,
            },
        )

        fig.add_annotation(
            text="     Rank",
            x=1.06,
            y=1.09,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": "black"},
            xanchor="center",
        )

        fig.update_layout(
            title={
                "text": f"Slice Plot of {param}",
                "x": 0.5,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
            },
            yaxis={
                "title": yaxis_title,
            },
        )

        fig.update_layout(dragmode=False)

        fig.write_html(
            file=f"{savepath}/{param}_slice_plot.html", config=standard_config
        )


class TickInfo(NamedTuple):
    coloridxs: list[float]
    text: list[str]


def get_tick_info(target_values: np.ndarray) -> TickInfo:
    sorted_target_values = np.sort(target_values)
    coloridxs = [0, 0.25, 0.5, 0.75, 1]
    values = np.quantile(sorted_target_values, coloridxs)
    rank_text = ["min.", "25%", "50%", "75%", "max."]
    text = [f"{rank_text[i]} ({values[i]:3g})" for i in range(len(values))]
    return TickInfo(coloridxs=coloridxs, text=text)
