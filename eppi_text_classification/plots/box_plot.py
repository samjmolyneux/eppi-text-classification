import numpy as np
import plotly.graph_objects as go


# TO DO: Get rid of the trace being at 0 before making open source.
def python_generate_box_plot_html(
    data_list, name_list, title, yaxis_title, xaxis_title
):
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
                fillcolor=fc,
                line_color=lc,
                boxpoints="all",
                marker={
                    "color": lc,
                    "line": {"color": "black", "width": 1},
                },
                hoverinfo="skip",
            )
        )

        original_fillcolours.append(fc)
        original_linecolours.append(lc)

    y_range = np.linspace(
        min([min(d) for d in data_list]) - 0.01,
        max([max(d) for d in data_list]) + 0.01,
        1000,
    )
    x_range = [0] * len(y_range)  # Dummy x-values for the invisible scatter trace
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            hoverinfo="y",  # Display only the y-value in hover info
            line={"color": "rgba(0,0,0,0)"},  # Make the line invisible
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis={
            "title": xaxis_title,
            "range": [-0.5, n - 0.5],  # a half-unit margin on each side
            "autorange": False,
            "zeroline": False,
        },
        yaxis={
            "title": yaxis_title,
            "showspikes": True,  # Enable spikes on the y-axis
            "spikemode": "across+toaxis",  # Spike line across the plot and to the axis
            "spikecolor": "black",
            "spikethickness": 1,
            "hoverformat": ".4f",  # Display 4 decimal places in the hover labels
        },
        hovermode="y",  # Enable hover on the y-axis
    )

    return fig


def box_plot(
    data_by_box,
    box_names,
    title,
    xaxis_title,
    yaxis_title,
    image_filename="box_plot",
    savepath="box_plot.html",
):
    # Generate the invisible scatter trace to get hoverline
    invisible_y = np.linspace(
        min([min(d) for d in data_by_box]),
        max([max(d) for d in data_by_box]),
        1000,
    ).tolist()

    invisible_x = [0] * len(invisible_y)

    data_by_box = [np.array(box_data).tolist() for box_data in data_by_box]

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Plotly Box Plot</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
            }}

            #box-plot-container {{
                width: 80%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 1rem;
            }}

            #box-plot-div {{
                width: 100%;
                aspect-ratio: 16 / 9;
            }}

        </style>
    </head>
    <body>
        <div id="box-plot-container">
            <div id="box-plot-div"></div>
        </div>

        <script>
            function createBoxPlot(dataList, nameList, title, xaxisTitle, yaxisTitle, divId) {{

                // Existing + MORE new colors
                const fillPalette = [
                //"rgba(0, 0, 255, 0.5)",    // Blue
                //"rgba(255, 0, 0, 0.5)",    // Red
                //"rgba(0, 200, 0, 0.5)",    // Green
                //"rgba(255, 165, 0, 0.5)",  // Orange
                //"rgba(128, 0, 128, 0.5)",  // Purple
                //"rgba(0, 255, 255, 0.5)",  // Cyan
                //"rgba(255, 105, 180, 0.5)",// Hot Pink
                //"rgba(255, 215, 0, 0.5)",  // Gold
                //"rgba(139, 69, 19, 0.5)",  // Brown
                //"rgba(154, 205, 50, 0.5)", // Yellow-Green
                //"rgba(0, 191, 255, 0.5)",  // Deep Sky Blue
                //"rgba(173, 255, 47, 0.5)", // Lime Green
                //"rgba(210, 105, 30, 0.5)", // Chocolate
                //"rgba(240, 230, 140, 0.5)",// Khaki
                //"rgba(70, 130, 180, 0.5)", // Steel Blue
                //"rgba(255, 140, 0, 0.5)",  // Dark Orange
                //"rgba(0, 250, 154, 0.5)",  // Medium Spring Green
                //"rgba(255, 0, 255, 0.5)",  // Magenta
                //"rgba(128, 128, 0, 0.5)",  // Olive
                //"rgba(0, 0, 128, 0.5)",    // Navy
                ];

                const linePalette = [
                "blue",
                "red",
                "green",
                "orange",
                "purple",
                "cyan",
                "hotpink",
                "gold",
                "brown",
                "yellowgreen",
                "deepskyblue",
                "limegreen",
                "chocolate",
                "khaki",
                "steelblue",
                "darkorange",
                "mediumspringgreen",
                "magenta",
                "olive",
                "navy",
                ];


                let traces = [];

                for (let i = 0; i < dataList.length; i++) {{
                    const fc = fillPalette[i % fillPalette.length];
                    const lc = linePalette[i % linePalette.length];

                    traces.push({{
                        type: "box",
                        y: dataList[i],
                        name: nameList[i],
                        fillcolor: fc,
                        line: {{ color: lc }},
                        boxpoints: "all",
                        marker: {{
                            color: lc,
                            line: {{ color: "black", width: 1 }}
                        }},
                        hoverinfo: "skip",
                    }});
                }}

                traces.push({{
                    type: "scatter",
                    x : {invisible_x},
                    y : {invisible_y},
                    mode: "lines",
                    hoverinfo: "y",
                    line: {{ color: "rgba(0,0,0,0)" }},
                    showlegend: false,
                }});

                const layout = {{
                    title: {{
                        text: title,
                        font: {{ size: 32}},
                        x:0.5,
                        xref: "paper",
                        xanchor: "center",
                    }},
                    showlegend: true,
                    xaxis: {{
                        range: [-0.5, dataList.length - 0.5],
                        autorange: false,
                        zeroline: false,
                        type: "category",
                    }},
                    yaxis: {{
                        title: {{
                            text: yaxisTitle,
                            standoff: 20,
                            font: {{ size: 20}},
                        }},
                        showspikes: true,
                        spikemode: "across+toaxis",
                        spikecolor: "black",
                        spikethickness: 1,
                        hoverformat: ".4f",
                    }},
                    hovermode: "y unified",
                    dragmode: false,


                }};

                const config = {{
                    responsive: true,
                    scrollZoom: false,
                    showLink: true,
                    plotlyServerURL: "https://chart-studio.plotly.com",
                    modeBarButtons: [["toImage"]],
                    displaylogo: false,
                    displayModeBar: "always",
                    toImageButtonOptions: {{
                        format: "png",
                        filename: "{image_filename}",
                        height: 720,
                        width: 1480,
                        scale: 3
                    }}
                }};

                Plotly.newPlot(divId, traces, layout, config);
            }}

            // Call the function with Python-generated data
            const dataList = {data_by_box};
            const nameList = {box_names};

            createBoxPlot(dataList, nameList, "{title}", "{xaxis_title}", "{yaxis_title}", "box-plot-div");
        </script>
    </body>
    </html>
    """

    # Save the HTML file
    with open(savepath, "w") as f:
        f.write(html_content)

    print(f"HTML file saved at: {savepath}")
