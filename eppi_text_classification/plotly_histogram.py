import numpy as np

def create_histogram_html(scores, savepath, title="", xaxis_title=""):
    # Generate the HTML file
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

        <script>
            var aucScores = {scores};  // Embed Python list into JavaScript

            function createHistogram(binSize) {{
                var trace = {{
                    x: aucScores,
                    type: "histogram",
                    marker: {{
                        color: "rgba(102, 204, 255, 0.7)",  // Softer blue with transparency
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


            function updateBins(value) {{
                Plotly.restyle("histogram", "xbins.size", [parseFloat(value)]);
            }}

            // Initial plot with default bin size
            createHistogram(0.005);
        </script>
    </body>
    </html>
    """

    # Save the HTML file
    with open(savepath, "w") as f:
        f.write(html_content)

    print(f"HTML file saved at {savepath}")