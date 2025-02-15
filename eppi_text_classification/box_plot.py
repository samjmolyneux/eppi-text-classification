import plotly.graph_objects as go
import numpy as np


def box_plot(data_list, name_list, title, y_axis_title, x_axis_title):

    fig = go.Figure()

    # A small colour palette. If data_list has more violins than this,
    # we'll cycle through the palette:
    fill_palette = [
        "rgba(0, 0, 255, 0.5)",
        "rgba(255, 0, 0, 0.5)",
        "rgba(0, 200, 0, 0.5)",
        "rgba(255, 165, 0, 0.5)",
        "rgba(128, 0, 128, 0.5)",
        "rgba(0, 255, 255, 0.5)",
    ]
    line_palette = ["blue", "red", "green", "orange", "purple", "cyan"]

    n = len(data_list)  # Number of violins

    # We will store each violin's *original* fill and line colours,
    # so we can restore them when toggling "Violin On":
    original_fillcolours = []
    original_linecolours = []

    for i, data in enumerate(data_list):
        fc = fill_palette[i % len(fill_palette)]  # pick fill colour
        lc = line_palette[i % len(line_palette)]  # pick line colour

        fig.add_trace(
            go.Box(
                y=data,
                name=name_list[i],
                # Violin shape
                fillcolor=fc,
                line_color=lc,
                # boxmean="sd",
                boxpoints="all",
                # Points
                # points="all",
                marker=dict(
                    color=lc,  # keep points in line-colour
                    line=dict(color="black", width=1),
                ),
                # Hover
                hoverinfo="skip",
            )
        )

        original_fillcolours.append(fc)
        original_linecolours.append(lc)

    y_range = np.linspace(min([min(d) for d in data_list]) - 0.01, max([max(d) for d in data_list]) + 0.01, 1000)
    x_range = [0] * len(y_range)  # Dummy x-values for the invisible scatter trace
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            hoverinfo="y",  # Display only the y-value in hover info
            line=dict(color="rgba(0,0,0,0)"),  # Make the line invisible
            showlegend=False,
        )
    ) 
    
    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis=dict(
            title = x_axis_title,
            range=[-0.5, n - 0.5],  # a half-unit margin on each side
            autorange=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=y_axis_title,
            showspikes=True,  # Enable spikes on the y-axis
            spikemode="across+toaxis",  # Spike line across the plot and to the axis
            spikecolor="black",
            spikethickness=1,
        ),
        hovermode="y",  # Enable hover on the y-axis
    )

    return fig
