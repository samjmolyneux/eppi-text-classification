import plotly.graph_objects as go
from sklearn.metrics import roc_curve
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification


def plotly_roc(y_test, y_test_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_probs)
    roc_auc = auc(fpr, tpr)

    df = pd.DataFrame(
        {"False Positive Rate": fpr, "True Positive Rate": tpr, "Threshold": thresholds}
    )
    df["Threshold"] = df["Threshold"].apply(lambda x: f"{x:.4g}")

    fig = px.area(
        df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC={roc_auc:.4f})",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        hover_data={"Threshold": True},
        width=600,
        height=600,
    )

    fig.add_shape(
        type="line", line=dict(dash="dash", color="grey"), x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    fig.update_layout(
        title={"text": f"ROC Curve (AUC={roc_auc:.4f})", "x": 0.5, "xanchor": "center"}
    )

    fig.show()
