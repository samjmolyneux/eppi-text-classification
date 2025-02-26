import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import average_precision_score, roc_auc_score


def select_threshold_plot(
    true_y,
    pred_scores,
    output_html_path="select_threshold_plot.html",
):
    # generate the X_N, X_P and stuff here
    x_N, y_N = _get_density_curve_data([pred_scores[true_y == 0]])
    x_P, y_P = _get_density_curve_data([pred_scores[true_y == 1]])

    roc_auc = roc_auc_score(true_y, pred_scores)
    pr_auc = average_precision_score(true_y, pred_scores)

    pred_scores = pred_scores.tolist()
    true_y = true_y.tolist()

    x_N = x_N.tolist()
    y_N = y_N.tolist()
    x_P = x_P.tolist()
    y_P = y_P.tolist()

    html_template = f"""<!DOCTYPE html>
  <html>
  <head>
    <meta charset="utf-8" />
    <title>Interactive Probabilities Density Plot (Pure JS)</title>
    <!-- Load Plotly from CDN -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>
      input[type="number"]::-webkit-outer-spin-button,
      input[type="number"]::-webkit-inner-spin-button {{
          -webkit-appearance: none;
          margin: 0;
      }}

      input[type="number"] {{
          -moz-appearance: textfield;
      }}
    </style>
  </head>

  <body>

  <div id="plotDiv" style="width: 95vw; height: 80vh; margin-top: 4vh;"></div>

  <div style="margin-top: 0em; display: flex; flex-direction: column; gap: 0.5em;">

    <div style="font-size: 1em;">
      Threshold: <span id="thresholdValue">0.5</span>
    </div>

    <input
      type="range"
      id="thresholdSlider"
      min="{min(pred_scores)}"
      max="{max(pred_scores)}"
      step="0.01"
      value="{min(pred_scores) + 0.5 * (max(pred_scores) - min(pred_scores))}"
      oninput="updatePlot(parseFloat(this.value));"
      style="width: 40%;"
    />

    <div style="display: flex; gap: 0.5em;">
      <label for="thresholdInput" style="font-size: 0.875em;">Set Threshold:</label>
      <input
        type="number"
        id="thresholdInput"
        min="{min(pred_scores)}"
        max="{max(pred_scores)}"
        step="0.01"
        value="{min(pred_scores) + 0.5 * (max(pred_scores) - min(pred_scores)):.4f}"
        style="width: 8em; text-align: left;"
        onfocus="clearAndStore(this);"
        onblur="restoreIfEmpty(this);"
      />
    </div>


    <div style="display: flex; gap: 0.5em;">
      <label for="recallInput" style="font-size: 0.875em;">Set Recall:</label>
      <input
        type="number"
        id="recallInput"
        min="{0}"
        max="{1}"
        step="0.01"
        style="width: 8em; text-align: left;"
        onfocus="clearAndStore(this)"
        onblur="restoreIfEmpty(this)"
      />
    </div>
  </div>


  <script>
  // --------------------------------------------------------------------
  // 1) DUMMY DATA (Replace these arrays with your actual data)
  // --------------------------------------------------------------------

  // For simplicity, let's assume we already computed xN,yN and xP,yP for
  // two kernel density estimates: negative (N) and positive (P).

  var xN = {x_N};
  var xP = {x_P};
  var yN = {y_N};
  var yP = {y_P};
  var predictedProba = {pred_scores};
  var trueY = {true_y};


  // For the "cumulative gains" subplot, we can define an array of x=proportionExamined
  // and y=cumulativeRecall, sorted by predictedProba descending:
  var sortedScoresIndices = predictedProba
    .map((value, index) => [value, index])
    .sort((a, b) => b[0] - a[0])
    .map(pair => pair[1]);

  var sortedTrueY = sortedScoresIndices.map(idx => trueY[idx]);
  var nPos        = sortedTrueY.reduce((acc,v)=>acc+v,0);  // total # of positives
  var cumulative  = [0];
  var sumPos      = 0;
  for(var i=0; i<sortedTrueY.length; i++){{
    sumPos += sortedTrueY[i];
    cumulative.push(sumPos / nPos);
  }}
  var xGains = [...Array(trueY.length+1).keys()].map(i => i/(trueY.length));
  var yGains = cumulative; // same length as xGains

  let thresholdBoundaries = [Infinity, ...predictedProba.slice()].sort((a,b)=>b-a);

  // We'll keep track of the maximum Y for the negative and positive distribution plots:
  var maxDistY = Math.max(...yN, ...yP);


  // --------------------------------------------------------------------
  // 2) FUNCTIONS TO COMPUTE SPLITS, METRICS, CONFUSION MATRIX, ETC.
  // --------------------------------------------------------------------

  // Quick function to find the index to split a sorted array x[] at value 'threshold'
  function findSplitIndex(xArr, threshold){{
    // We want the first index where xArr[index] >= threshold
    // (similar to Python's np.searchsorted)
    for(let i=0; i<xArr.length; i++){{
      if(xArr[i] >= threshold){{
        return i;
      }}
    }}
    return xArr.length; // if threshold is bigger than all
  }}

  // Compute confusion matrix [TN, FP, FN, TP], plus metrics
  function computeConfusionMatrixAndMetrics(threshold){{
    let TP=0, TN=0, FP=0, FN=0;
    for(let i=0; i<predictedProba.length; i++){{
      let predLabel = (predictedProba[i] >= threshold) ? 1 : 0;
      let actual    = trueY[i];
      if(actual===0 && predLabel===0){{ TN++; }}
      if(actual===0 && predLabel===1){{ FP++; }}
      if(actual===1 && predLabel===0){{ FN++; }}
      if(actual===1 && predLabel===1){{ TP++; }}
    }}
    let accuracy  = (TP + TN)/(TP + TN + FP + FN);
    let TPR       = (TP + FN>0) ? (TP/(TP+FN)) : 0;  // recall
    let FPR       = (FP + TN>0) ? (FP/(FP+TN)) : 0;
    let precision = (TP + FP>0) ? (TP/(TP+FP)) : 0;
    let balancedAccuracy = 0.5 * (TPR + (1 - FPR));

    // F4 = (1 + 4^2) * precision * recall / (4^2 * precision + recall)
    //     = 17*precision*TPR / (16*precision + TPR) (from your code)
    let F4 = (17 * precision * TPR)/(16*precision + TPR || 1); // avoid /0

    return {{
      TN, FP, FN, TP,
      accuracy, TPR, FPR, precision, balancedAccuracy, F4
    }};
  }}

  // --------------------------------------------------------------------
  // 3) BUILD INITIAL FIGURE (SUBPLOTS, TRACES, LAYOUT)
  //    - We'll create a single set of traces (TN, FP, FN, TP, threshold lines, etc.)
  //    - Then we update them in 'updatePlot()' rather than making separate frames.
  // --------------------------------------------------------------------

  //
  // Confusion Matrix Heatmap: We'll keep the same z, but we only use it for color
  // We'll place text for [FP, TN, TP, FN] dynamically. (z = [[0,1],[2,3]] per your code)
  //
  var confusionZ = [
    [0, 1],
    [2, 3]
  ];
  var confusionColorscale = [
    [0.00, "pink"],
    [0.25, "pink"],
    [0.25, "salmon"],
    [0.50, "salmon"],
    [0.50, "lightgreen"],
    [0.75, "lightgreen"],
    [0.75, "green"],
    [1.00, "green"]
  ];

  // Invariant metrics table (ROC AUC, PR AUC, etc.)
  // We'll show them in a table trace. You can dynamically compute them in JS or
  // precompute them in Python and embed them. For demonstration, let's just fix them.
  var invariantHeader = [["Invariant Metric","Value"]];
  var invariantRows   = [
    ["ROC AUC", {roc_auc:.4f}],
    ["PR AUC", {pr_auc:.4f}]
  ];
  var invariantCellsValues = [
    invariantRows.map(r => r[0]),
    invariantRows.map(r => r[1])
  ];

  // We also need a "variant" metrics table (Accuracy, Balanced Accuracy, etc.)
  // We'll start with placeholders; we will restyle them on each updatePlot().
  var variantHeader = [["Metric","Value"]];
  var variantRows   = [
    ["Accuracy",         "--"],
    ["Balanced Accuracy","--"],
    ["Recall",           "--"],
    ["Precision",        "--"],
    ["FPR",              "--"],
    ["F4",               "--"]
  ];
  var variantCellsValues = [
    variantRows.map(r => r[0]),
    variantRows.map(r => r[1])
  ];

  // Create initial traces (most are placeholders; we’ll fill them on update):

  // 3.1 Negative distribution splits: TN (left side) and FP (right side)
  var traceTN = {{
    x: [],
    y: [],
    mode: 'lines',
    fill: 'tozeroy',
    line: {{color: 'salmon'}},
    name: 'TN',
    legendgroup: 'Negative',
    xaxis: 'x',  // top-left subplot
    yaxis: 'y',
    hoverinfo: 'skip',
  }};
  var traceFP = {{
    x: [],
    y: [],
    mode: 'lines',
    fill: 'tozeroy',
    line: {{color: 'pink'}},
    name: 'FP',
    legendgroup: 'Negative',
    xaxis: 'x',
    yaxis: 'y',
    hoverinfo: 'skip',
  }};

  // 3.2 Positive distribution splits: FN (left side) and TP (right side)
  var traceFN = {{
    x: [],
    y: [],
    mode: 'lines',
    fill: 'tozeroy',
    line: {{color: 'green'}},
    name: 'FN',
    legendgroup: 'Positive',
    xaxis: 'x3', // bottom-left subplot
    yaxis: 'y3',
    hoverinfo: 'skip'
  }};
  var traceTP = {{
    x: [],
    y: [],
    mode: 'lines',
    fill: 'tozeroy',
    line: {{color: 'lightgreen'}},
    name: 'TP',
    legendgroup: 'Positive',
    xaxis: 'x3',
    yaxis: 'y3',
    hoverinfo: 'skip',
  }};

  // 3.3 Vertical threshold lines on top-left and bottom-left distribution subplots
  var threshLineNegative = {{
    x: [], // e.g. [threshold, threshold]
    y: [], // e.g. [0, maxDistY]
    mode: 'lines',
    line: {{dash: 'dash', color: '#20313e'}},
    showlegend: false,
    xaxis: 'x',
    yaxis: 'y',
    hoverinfo: 'skip',
  }};
  var threshLinePositive = {{
    x: [],
    y: [],
    mode: 'lines',
    line: {{dash: 'dash', color: '#20313e'}},
    showlegend: false,
    xaxis: 'x3',
    yaxis: 'y3',
    hoverinfo: 'skip',
  }};

  // 3.4 Confusion matrix heatmap (top-right)
  var confusionTrace = {{
    z: confusionZ,
    x: ['0','1'],   // predicted label
    y: ['0','1'],   // true label
    type: 'heatmap',
    colorscale: confusionColorscale,
    showscale: false,
    xaxis: 'x2',
    yaxis: 'y2',
    hoverinfo: 'skip',
  }};

  // 3.5 Invariant metrics table (just one table trace). Domain set so it's in top-right quadrant
  var invariantTableTrace = {{
    type: 'table',
    header: {{
      values: invariantHeader[0],
      fill: {{color: "lightgrey"}},
      align: "center"
    }},
    cells: {{
      values: invariantCellsValues,
      fill: {{color: "white"}},
      align: "center"
    }},
    domain: {{
      x: [0.78, 1.0],
      y: [0.475, 0.675]
    }}
  }};

  // 3.6 "Variant" metrics table (Accuracy, etc.) - also in top-right quadrant
  var variantTableTrace = {{
    type: 'table',
    header: {{
      values: variantHeader[0],
      fill: {{color: "lightgrey"}},
      align: "center"
    }},
    cells: {{
      values: variantCellsValues,
      fill: {{color: "white"}},
      align: "center"
    }},
    domain: {{
      x: [0.78, 1.0],
      y: [0.68, 1.0]
    }}
  }};

  // 3.7 Cumulative gains line (bottom-right)
  var gainsLine = {{
    x: xGains,
    y: yGains,
    mode: 'lines',
    name: "",
    line: {{width: 3, color: '#20C5FF'}},
    // hovertemplate: 'Examined: %{{x:.2f}}<br>Recall: %{{y:.2f}}',
    showlegend: false,
    xaxis: 'x4',
    yaxis: 'y4',

    customdata:thresholdBoundaries,
    hovertemplate:(
        "Proportion of Papers Examined: %{{x}}"
        + "<br>Proportion of Positives Found: %{{y:.4f}}"
        + "<br>Decision Threshold: %{{customdata:.4f}}"
    ),
    hoverlabel:{{
        bgcolor:"white",
        bordercolor:"#20C5FF",
        font: {{color: "black"}},
    }},
    legendgroup:"",
  }};

  // 3.8 Two lines to mark threshold on the gains plot
  //     We'll place them at (pAT, 0) -> (pAT, TPR) and (0, TPR) -> (pAT, TPR)
  var gainsVLine = {{
    x: [],
    y: [],
    mode: 'lines',
    line: {{color: 'grey'}},
    showlegend: false,
    xaxis: 'x4',
    yaxis: 'y4',
    name:"",
  }};
  var gainsHLine = {{
    x: [],
    y: [],
    mode: 'lines',
    line: {{color: 'grey'}},
    showlegend: false,
    xaxis: 'x4',
    yaxis: 'y4',
    name: "",
  }};

  let minProba = Math.min(...predictedProba);
  let maxProba = Math.max(...predictedProba);
  let rangeProba = maxProba - minProba;

  // The entire data array for the initial plot:
  var data = [
    // top-left negative distribution
    traceTN, traceFP, threshLineNegative,
    // bottom-left positive distribution
    traceFN, traceTP, threshLinePositive,
    // top-right confusion matrix
    confusionTrace,
    // top-right tables
    invariantTableTrace,
    variantTableTrace,
    // bottom-right gains line + threshold lines
    gainsLine, gainsVLine, gainsHLine
  ];

  // Layout with four subplots arranged:
  //   row=1 col=1 => negative distribution   (xaxis='x',   yaxis='y')
  //   row=1 col=2 => confusion matrix + table(xaxis='x2',  yaxis='y2')
  //   row=2 col=1 => positive distribution   (xaxis='x3',  yaxis='y3')
  //   row=2 col=2 => gains plot              (xaxis='x4',  yaxis='y4')
  var layout = {{
    //title: {{
    //  text: "<b>Choose your threshold</b><br>",
    //  y: 0.97,
    //  yanchor: 'bottom'
    //}},
    margin: {{
      // l: 20,
      // r: 20,
      b: 40,
      t: 20,
    }},
    grid: {{
      rows: 2,
      columns: 2,
      pattern: 'independent', // each cell has its own axes
      roworder: 'top to bottom'
    }},
    // Domain definitions for each subplot (optional, but shown for clarity)
    xaxis: {{
      domain: [0.0, 0.45], // top-left
      anchor: 'y',
      // autorange: false,
      range: [
        minProba - 0.05*rangeProba,
        maxProba + 0.05*rangeProba,
      ],
      zeroline: false,
    }},
    yaxis: {{
      domain: [0.55,1.0],  // top-left
      anchor: 'x',
      title: 'Actual Negatives',
      range: [0, maxDistY*1.1],
      // zeroline: false,
    }},
    xaxis2: {{
      domain: [0.55, 0.775], // top-right (heatmap only)
      anchor: 'y2',
      showgrid: false,
      zeroline: false,
      showspikes: false,
      showticklabels: false,
      ticks: '',
    }},
    yaxis2: {{
      domain: [0.55, 1.0],  // top-right
      anchor: 'x2',
      showgrid: false,
      zeroline: false,
      showspikes: false,
      showticklabels: false,
      ticks: '',
    }},
    xaxis3: {{
      domain: [0.0, 0.45],  // bottom-left
      anchor: 'y3',
      title: 'Predicted Probability',
      // autorange: false,
      range: [
        minProba - 0.05*rangeProba,
        maxProba + 0.05*rangeProba,
      ],
      zeroline: false,
    }},
    yaxis3: {{
      domain: [0.0, 0.45],
      anchor: 'x3',
      title: 'Actual Positives',
      range: [0, maxDistY*1.1]
    }},
    xaxis4: {{
      domain: [0.55, 1.0],  // bottom-right
      anchor: 'y4',
      title: 'Proportion of Papers Examined',
      range: [-0.02, 1.02]
    }},
    yaxis4: {{
      domain: [0.0, 0.45],
      anchor: 'x4',
      title: 'Recall',
      range: [-0.02, 1.02]
    }},
    annotations: [],
    shapes: [
      // The confusion matrix boundary lines
      {{
        type: "rect", xref: "x2", yref: "y2",
        x0: -0.5, y0: -0.5, x1: 1.5, y1: 1.5,
        line: {{color: "black", width: 2}}
      }},
      {{
        type: "line", xref: "x2", yref: "y2",
        x0: -0.5, y0: 0.5, x1: 1.5, y1: 0.5,
        line: {{color: "black", width: 1.5}}
      }},
      {{
        type: "line", xref: "x2", yref: "y2",
        x0: 0.5, y0: -0.5, x1: 0.5, y1: 1.5,
        line: {{color: "black", width: 1.5}}
      }}
    ],
    dragmode: false,
  }};

  var config = {{
    responsive: true,
    scrollZoom: false,
    modeBarButtons: [["toImage"]],
    displaylogo: false,
    displayModeBar: "always",
    toImageButtonOptions: {{
      format: "png",
      filename: "eppi-select-threshold",
      height: 720,
      width: 1480,
      scale: 3
    }}
  }};
  // Make the initial plot
  Plotly.newPlot('plotDiv', data, layout, config);

  // --------------------------------------------------------------------
  // 4) THE KEY FUNCTION: updatePlot(threshold)
  //    This slices the distributions, updates confusion matrix, tables, etc.
  // --------------------------------------------------------------------
  function updatePlot(threshold){{
    // Display threshold above slider


    // 1) Find the split index for negative distribution
    var idxN = findSplitIndex(xN, threshold);
    //   TN = xN[0..idxN-1], yN[0..idxN-1]
    //   FP = xN[idxN..end], yN[idxN..end]
    if (idxN > 0) {{
      let xNDiff = xN[idxN] - xN[idxN-1];
      let yNDiff = yN[idxN] - yN[idxN-1];
      let yInterpolated = yN[idxN] +  ((threshold - xN[idxN])*yNDiff)/xNDiff;

      var xTN = [...xN.slice(0, idxN), threshold];
      var yTN = [...yN.slice(0, idxN), yInterpolated];
      var xFP = [threshold, ...xN.slice(idxN)];
      var yFP = [yInterpolated, ...yN.slice(idxN)];

    }} else {{
      var xTN = [];
      var yTN = [];
      var xFP = xN.slice()
      var yFP = yN.slice()
    }}
    // var xTN = [...xN.slice(0, idxN), threshold];
    // var yTN = [...yN.slice(0, idxN), ;
    // var xFP = [threshold, ...xN.slice(idxN)];
    // var yFP = yN.slice(idxN);

    // 2) Similarly for positive distribution
    var idxP = findSplitIndex(xP, threshold);

    if (idxP > 0){{
      let xPDiff = xP[idxP] - xP[idxP-1];
      let yPDiff = yP[idxP] - yP[idxP-1];
      let yInterpolated = yP[idxP] +  ((threshold - xP[idxP])*yPDiff)/xPDiff;

      var xFN = [...xP.slice(0, idxP), threshold];
      var yFN = [...yP.slice(0, idxP), yInterpolated];
      var xTP = [threshold, ...xP.slice(idxP)];
      var yTP = [yInterpolated, ...yP.slice(idxP)];

    }} else {{
      var xFN = [];
      var yFN = [];
      var xTP = xP.slice();
      var yTP = yP.slice();
    }}

    // 3) Vertical dashed lines on distributions
    var lineNegX = [threshold, threshold];
    var lineNegY = [0, maxDistY*1.1];
    var linePosX = [threshold, threshold];
    var linePosY = [0, maxDistY*1.1];

    var cm = computeConfusionMatrixAndMetrics(threshold);
    var matrixText = [
      [`FP: ${{cm.FP}}`, `TN: ${{cm.TN}}`],
      [`TP: ${{cm.TP}}`, `FN: ${{cm.FN}}`]
    ];

    if (threshold > minProba + 0.96 * rangeProba){{
      TNVisible = true;
      FNVisible = true;
      FPVisible = false;
      TPVisible = false;
    }} else if (threshold < minProba + 0.04 * rangeProba){{
      TNVisible = false;
      FNVisible = false;
      FPVisible = true;
      TPVisible = true;
    }} else {{
      TNVisible = true;
      FNVisible = true;
      FPVisible = true;
      TPVisible = true;
    }}


    // 5) "Variant" table metrics:
    var accuracy         = (100*cm.accuracy).toFixed(4) + '%';
    var balancedAccuracy = (100*cm.balancedAccuracy).toFixed(4) + '%';
    var recall           = (100*cm.TPR).toFixed(4) + '%';
    var precision        = (100*cm.precision).toFixed(4) + '%';
    var specificity      = (100*(1-cm.FPR)).toFixed(4) + '%';
    var fpr              = (100*cm.FPR).toFixed(4) + '%';
    var f4               = (100*cm.F4).toFixed(4) + '%';

    var newVariantValues = [
      ["Accuracy", "Balanced Accuracy", "Recall", "Precision", "Specificity", "FPR", "F4"],
      [accuracy, balancedAccuracy, recall, precision, specificity, fpr, f4]
    ];

    // 6) Gains plot threshold lines
    var proportionAbove = predictedProba.filter(p => p >= threshold).length / predictedProba.length;
    var gainsVx = [proportionAbove, proportionAbove];
    var gainsVy = [0, cm.TPR];
    var gainsHx = [0, proportionAbove];
    var gainsHy = [cm.TPR, cm.TPR];

    // 7) Send a single update to Plotly for everything that changes:
    //    We'll do an array of "restyle" updates for each trace, plus "relayout" updates for the heatmap text.
    //    The order of the traces in data[]:
    //      0: traceTN, 1: traceFP, 2: threshLineNegative,
    //      3: traceFN, 4: traceTP, 5: threshLinePositive,
    //      6: confusionTrace,
    //      7: invariantTableTrace, 8: variantTableTrace,
    //      9: gainsLine, 10: gainsVLine, 11: gainsHLine
    //
    const newAnnotations = [
      {{
        // The "FP" cell is at x='0', y='0' in the heatmap data space
        text: `FP: ${{cm.FP}}`,
        x: '0',
        y: '0',
        xref: 'x2',  // x-axis for your heatmap
        yref: 'y2',  // y-axis for your heatmap
        showarrow: false,
        font: {{ color: 'white', size: 14 }}
      }},
      {{
        // The "TN" cell is at x='1', y='0'
        text: `TN: ${{cm.TN}}`,
        x: '1',
        y: '0',
        xref: 'x2',
        yref: 'y2',
        showarrow: false,
        font: {{ color: 'white', size: 14 }}
      }},
      {{
        // The "TP" cell is at x='0', y='1'
        text: `TP: ${{cm.TP}}`,
        x: '0',
        y: '1',
        xref: 'x2',
        yref: 'y2',
        showarrow: false,
        font: {{ color: 'white', size: 14 }}
      }},
      {{
        // The "FN" cell is at x='1', y='1'
        text: `FN: ${{cm.FN}}`,
        x: '1',
        y: '1',
        xref: 'x2',
        yref: 'y2',
        showarrow: false,
        font: {{ color: 'white', size: 14 }}
      }},
      // Annotation of threshold line
      {{
        text:"Predicted Negative        Predicted Positive",
        y:0.5,
        yref:"paper",
        x:threshold,
        visible:true,
        // font:{{size:14}},
        showarrow:false,
      }},
      {{
        text:"TN",
        y:0.97,
        yref:"y domain",
        x: minProba + 0.025 * rangeProba,
        visible:TNVisible,
        font:{{size:14}},
        showarrow:false,
      }},
      {{
        text:"FP",
        y:0.97,
        yref:"y domain",
        x: minProba + 0.975 * rangeProba,
        visible:FPVisible,
        font:{{size:14}},
        showarrow:false,
      }},
      {{
        text:"FN",
        y:0.97,
        yref:"y3 domain",
        x: minProba + 0.025 * rangeProba,
        visible:FNVisible,
        font:{{size:14}},
        showarrow:false,
      }},
      {{
        text:"TP",
        y:0.97,
        yref:"y3 domain",
        x: minProba + 0.975 * rangeProba,
        visible:TPVisible,
        font:{{size:14}},
        showarrow:false,
      }},
      {{
        text: proportionAbove.toFixed(4),
        x: proportionAbove,
        xref: "x4 domain",
        yref: "y4 domain",
        y: -0.06,
        font:{{size:12}},
        showarrow:false,
        bgcolor: "white",
        bordercolor: "black",
        borderwidth: 1,
        borderpad: 3,
      }},
      {{
        text: cm.TPR.toFixed(4),
        x: -0.07,
        xref: "x4 domain",
        yref: "y4 domain",
        y: cm.TPR.toFixed(2),
        font:{{size:12}},
        showarrow:false,
        bgcolor: "white",
        bordercolor: "black",
        borderwidth: 1,
        borderpad: 3,
      }}

    ];

    Plotly.restyle('plotDiv', {{
      x: [ xTN, xFP, lineNegX, xFN, xTP, linePosX, gainsVx, gainsHx],
      y: [ yTN, yFP, lineNegY, yFN, yTP, linePosY, gainsVy, gainsHy]
    }}, [0,1,2,3,4,5,  /* xTN, xFP, etc. go to these trace indices */
        10,11 // for x and y of gains lines, but we handle them in the same .restyle call
    ]);

    // Plotly.restyle('plotDiv', {{
    //   text: [[matrixText[0], matrixText[1]]],
    //   hoverinfo: [['text']]
    // }}, [6]);  // trace index 6 is confusionTrace

    Plotly.restyle('plotDiv',{{
      'cells.values': [ [newVariantValues[0], newVariantValues[1]] ]
    }}, [8]); // trace index 8 is variantTableTrace

    Plotly.relayout('plotDiv', {{
      annotations: newAnnotations
    }});

    document.getElementById("recallInput").value = (100*cm.TPR).toFixed(4);
    document.getElementById("thresholdInput").value = threshold.toFixed(4);
    document.getElementById("thresholdSlider").value = threshold.toFixed(4);
    document.getElementById('thresholdValue').textContent = threshold.toFixed(4);
  }}

  function syncThreshold(source) {{
    let input = document.getElementById("thresholdInput");

    updatePlot(parseFloat(input.value));
  }}

  updatePlot( parseFloat(document.getElementById('thresholdSlider').value) );

  document.getElementById("thresholdInput").addEventListener("keydown", function(event) {{
      if (event.key === "Enter") {{  // Only triggers when Enter is pressed in the input box
          syncThreshold("input");
      }}
  }});


  document.getElementById("recallInput").addEventListener("keydown", function(event) {{
      if (event.key === "Enter") {{  // Only triggers when Enter is pressed in the input box
          console.log("here")
          for (let i = cumulative.length-2; i >= 0; i--) {{
              if (cumulative[i]*100 < parseFloat(document.getElementById("recallInput").value)) {{
                  updatePlot(thresholdBoundaries[i+1]);
                  break;
              }}
          }}
      }}
  }});

  function clearAndStore(inputElement) {{
    inputElement.dataset.originalValue = inputElement.value;
    inputElement.value = '';
  }}

  function restoreIfEmpty(inputElement) {{
    if (inputElement.value === '') {{
      inputElement.value = inputElement.dataset.originalValue;
    }}
  }}

  </script>

  </body>
  </html>
  """
    with open(output_html_path, "w") as file:
        file.write(html_template)

    print(f"Generated HTML saved to {output_html_path}")


def _get_density_curve_data(data, curve_type="kde"):
    """
    Compute distribution data using plotly figure_factory distplot, to plot custom interactive density curve:

    Parameters
    ----------
    data: list containing a sequence of floats
        data of which the density curve will be computed
        e.g. list([0.5, 0.8, 0.6]) or [np.array([2.0, 5.8, 0.0])]
    curve_type: {'kde', 'normal'},  default=kde
        type of curve, either kernel density estimation or normal curve

    Returns
    -------
    x_dist_data: np.array
        array with x coordinates data for the density curve
    y_dist_data: np.array
        array with y coordinates data for the density curve

    """
    fig = ff.create_distplot(
        data,
        ["data"],
        show_hist=False,
        show_rug=False,
        curve_type=curve_type,
    )

    x_dist_data = np.array(fig["data"][0]["x"])
    y_dist_data = np.array(fig["data"][0]["y"])

    return x_dist_data, y_dist_data
