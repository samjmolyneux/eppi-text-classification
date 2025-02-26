import re

import optuna
import plotly
from optuna.visualization._rank import _get_rank_info, _get_tick_info

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


def create_slice_plot_html(param, study, savepath, yaxis_title="ROC_AUC"):
    info = _get_rank_info(study, [param], target=None, target_name="Objective Value")
    tick_info = _get_tick_info(info.zs)

    tick_labels = make_tick_labels_4f(tick_info.text)

    fig = optuna.visualization.plot_slice(study, params=[param])

    fig.update_traces(
        marker={
            "color": list(map(plotly.colors.label_rgb, info.colors)),
            "line": {"width": 0.5, "color": "Grey"},
            "colorbar": {
                "thickness": 25,
                "tickvals": tick_info.coloridxs,
                "ticktext": tick_labels,
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

    fig.write_html(file=savepath, config=standard_config)


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

    for param in study.best_params:
        create_slice_plot_html(param, study, f"{save_dir}/{param}_slice.html")
