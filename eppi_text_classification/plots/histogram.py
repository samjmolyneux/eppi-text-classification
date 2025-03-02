import numpy as np


def histogram_plot(
    scores, savepath, title="", xaxis_title="", colour="rgba(102, 204, 255, 0.7)"
):
    mean = np.mean(scores)
    median = np.median(scores)
    std_dev = np.std(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)

    scores = np.array(scores).tolist()

    # Generate the HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Histogram</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                margin-top: 2em;
            }}
            #histogram-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            #histogram {{
                width: 90%;
                max-width: 50rem;
                // margin-bottom: 0.5em;  /* Adds spacing before dropdown */
                aspect-ratio: 3/2;
            }}
            #binSelect {{
                font-size: 0.8rem;
                //padding: 0.3em;
            }}
            #bin-container {{
                width: 80%;  /* Same width as the histogram */
                max-width: 50rem;
                text-align: left;  /* Aligns text to the left */
                margin-left: auto; /* Centers container but keeps text left-aligned */
                margin-right: auto;
            }}

            #table-container {{
                display: flex;
                flex-direction: column;
                align-items: flex-start; /* Aligns the table with the left side */
                justify-content: flex-start;
                width: 80%;
                max-width: 50rem;
                margin-left: auto;
                margin-right: auto;
                margin-top: 2em; /* Adds spacing between histogram and table */
            }}

            #stats-table {{
                width:20rem;
                max-width: 100%; /* Ensures the table takes full width of container */
            }}

        </style>
    </head>
    <body>

        <div id="histogram-container">
            <div id="histogram"></div>
        </div>

        <div id="bin-container">
            <label for="binSelect">Select the bin size:</label>
            <select id="binSelect" onchange="updateBins(this.value)">
                <option value="0.0001">0.0001</option>
                <option value="0.0005">0.0005</option>
                <option value="0.001">0.001</option>
                <option value="0.002">0.002</option>
                <option value="0.005" selected>0.005</option>
                <option value="0.01">0.01</option>
                <option value="0.02">0.02</option>
            </select>
        </div>

        <div id="table-container">
            <div id="stats-table"></div>
        </div>

        <script>
            var aucScores = {scores};  // Embed Python list into JavaScript

            function createHistogram(binSize) {{
                var trace = {{
                    x: aucScores,
                    type: "histogram",
                    marker: {{
                        color: "{colour}",  // Softer blue with transparency
                        line: {{color: "rgba(0, 0, 0, 0.7)", width: 1.5}}
                    }},
                    opacity: 0.8,
                    xbins: {{ size: binSize }}
                }};

                var layout = {{
                    title: {{
                        text: "{title}",
                        font : {{size: 30}},
                    }},
                    xaxis: {{
                        title: {{
                            text: "{xaxis_title}",
                            font: {{ size: 16 }},
                        }},
                    }},
                    yaxis: {{
                        title: {{
                            text: "Count",
                            font: {{ size: 16 }},
                        }},
                    }},
                    template: "plotly_white",
                    dragmode: false,
                }};

                var config = {{
                    responsive: true,
                    scrollZoom: false,
                    showLink: true,
                    plotlyServerURL: "https://chart-studio.plotly.com",
                    modeBarButtons: [["toImage"]],
                    displaylogo: false,
                    displayModeBar: "always",
                    toImageButtonOptions: {{
                        format: "png",
                        filename: "eppi-histogram",
                        height: 600,
                        width: 900,
                        scale: 3
                    }}
                }};

                Plotly.newPlot("histogram", [trace], layout, config);
            }}


            function createStatsTable() {{
                let mean = {mean};
                let median = {median};
                let stdDev = {std_dev};
                let minVal = {min_val};
                let maxVal = {max_val};

                var statsTable = {{
                    type: "table",
                    header: {{
                        values: [["<b>Statistic</b>"], ["<b>Value</b>"]],
                        align: "center",
                        fill: {{color: "lightgrey"}}
                    }},
                    cells: {{
                        values: [
                            ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum"],
                            [mean.toFixed(4), median.toFixed(4), stdDev.toFixed(4), minVal.toFixed(4), maxVal.toFixed(4)]
                        ],
                        align: "center"
                    }}
                }};

                var layout = {{
                    title: {{
                        font : {{size: 24}},
                    }},
                    margin: {{ t: 10, b: 10, r:0, l:0 }},
                    template: "plotly_white"
                }};

                config = {{
                    displayModeBar: false,
                }}
                Plotly.newPlot("stats-table", [statsTable], layout, config);
            }}

            function updateBins(value) {{
                Plotly.restyle("histogram", "xbins.size", [parseFloat(value)]);
            }}

            // Initial plot with default bin size
            createHistogram(0.005);
            createStatsTable();

        </script>
    </body>
    </html>
    """

    # Save the HTML file
    with open(savepath, "w") as f:
        f.write(html_content)

    print(f"HTML file saved at {savepath}")


def positive_negative_scores_histogram_plot(y_true, pred_scores, savepath):
    negative_scores = pred_scores[y_true == 0].tolist()
    positive_scores = pred_scores[y_true == 1].tolist()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive AUC Histogram</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                margin-top: 2em;
            }}

            #histogram-container {{
                // display: flex;
                // flex-direction: column;
                // align-items: center;
                // justify-content: center;
                position: relative;
                width: 80%;
                margin: 0 auto;
                aspect-ratio: 16/9;
            }}

            #histogram {{
                //width: 80%;
               // //max-width: 50rem;
                // margin-bottom: 0.5em;  /* Adds spacing before dropdown */
                //aspect-ratio: 16/9;
                width: 100%;
                height: 100%;

            }}

            #tips-container {{
                position: absolute;  /* Overlay on top of #histogram */
                bottom: 10%;           /* Adjust spacing from the top */
                right: 1%;         /* Adjust spacing from the right */
                background-color: #f8f8f8;
                border-radius: 0.5rem;
                font-size: 0.7rem;

            }}

            #tips-container ul {{
                list-style-type: none;
                padding: 0.4rem;
                margin: 0
            }}

            #tips-container li {{
                padding: 0.3rem 0;
            }}
            #binSelect {{
                font-size: 0.8rem;
                //padding: 0.3em;
            }}
            #bin-container {{
                width: 80%;  /* Same width as the histogram */
                // max-width: 50rem;
                text-align: left;  /* Aligns text to the left */
                margin-left: auto; /* Centers container but keeps text left-aligned */
                margin-right: auto;
            }}

        </style>
    </head>
    <body>

        <div id="histogram-container">
            <div id="histogram"></div>

            <div id="tips-container">
                <ul>
                    <li>🔍 Scroll to Zoom</li>
                    <li>✋ Click & Move to Pan</li>
                    <li>🔄 Double-Click to Reset</li>
                </ul>
            </div>
        </div>

        <div id="bin-container">
            <label for="binSelect">Select the bin size:</label>
            <select id="binSelect" onchange="updateBins(this.value)">
                <option value="0.001">0.001</option>
                <option value="0.002">0.002</option>
                <option value="0.005">0.005</option>
                <option value="0.01">0.01</option>
                <option value="0.02">0.02</option>
                <option value="0.05">0.05</option>
                <option value="0.1" selected>0.1</option>
                <option value="0.2">0.2</option>
                <option value="0.5">0.5</option>
                <option value="1">1</option>
            </select>
        </div>

         <script>

            function createHistogram(binSize) {{
                var negativeTrace = {{
                    name: "Negative Class",
                    x: {negative_scores},
                    type: "histogram",
                    marker: {{
                        color: "rgba(255, 0, 0, 0.4)",
                        line: {{color: "rgba(255, 0, 0, 0.4)", width: 1.5}}
                    }},
                    xbins: {{ size: binSize }}
                }};

                var positiveTrace = {{
                    name: "Positive Class",
                    x: {positive_scores},
                    type: "histogram",
                    marker: {{
                        color: "rgba(0, 188, 0, 0.65)",
                        line: {{color: "rgba(0, 188, 0, 0.8)", width: 1.5}}
                    }},
                    xbins: {{ size: binSize }}
                }};


                var layout = {{
                    title: {{
                        text: "Predicted Scores on Positive and Negative Test Datasets",
                        font : {{size: 20}},
                    }},
                    xaxis: {{
                        title: {{
                            text: "Predicted Score",
                            font: {{ size: 16 }},
                        }},
                    }},
                    yaxis: {{
                        title: {{
                            text: "Count",
                            font: {{ size: 16 }},
                        }},
                    }},
                    template: "plotly_white",
                    barmode: "overlay",
                    dragmode: "pan",
                }};

                var config = {{
                    responsive: true,
                    scrollZoom: true,
                    showLink: true,
                    plotlyServerURL: "https://chart-studio.plotly.com",
                    modeBarButtons: [["toImage", "zoom2d", "pan2d", "autoScale2d"]],
                    displaylogo: false,
                    displayModeBar: "always",
                    toImageButtonOptions: {{
                        format: "png",
                        filename: "to-scale-pos-neg-histogram",
                        height: 720,
                        width: 1280,
                        scale: 3
                    }}
                }};

                Plotly.newPlot("histogram", [positiveTrace, negativeTrace], layout, config);
            }}


            function updateBins(value) {{
                Plotly.restyle("histogram", "xbins.size", [parseFloat(value)]);
            }}

            // Initial plot with default bin size
            createHistogram(0.1);

        </script>
    </body>
    </html>
    """

    # Save the HTML file
    with open(savepath, "w") as f:
        f.write(html_content)

    print(f"HTML file saved at {savepath}")
