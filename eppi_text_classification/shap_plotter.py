import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np


class ShapPlotter:
    def __init__(
        self,
        model,
        X_test,
        feature_names,
        subsample=None,
        tree_path_dependent=False,
    ):
        self.model = model
        self.X_test = X_test.toarray()
        self.feature_names = feature_names

        self.background_data = np.zeros(shape=(1, X_test.shape[-1]))

        if isinstance(model, LGBMClassifier):
            if not tree_path_dependent:
                self.explainer = shap.TreeExplainer(
                    model,
                    data=self.background_data,
                    model_output="probability",
                )
            else:
                self.explainer = shap.TreeExplainer(
                    model,
                    model_output="raw",
                    feature_perturbation="tree_path_dependent",
                )

        if subsample is not None:
            self.X_test = shap.sample(self.X_test, subsample)

        self.shap_values = self.explainer.shap_values(self.X_test)

    def dot_plot(self, num_display=10, log_scale=False):
        if isinstance(self.model, LGBMClassifier):
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                plot_type="dot",
                max_display=num_display,
                use_log_scale=log_scale,
                plot_size=(500, 10),
            )

    def bar_chart(self, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=num_display,
            )

    def violin_plot(self, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                plot_type="violin",
                max_display=num_display,
            )

    # TO DO: Double check that the use of threshold is right for the base value
    def decision_plot(self, threshold, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            shap.decision_plot(
                threshold,
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                feature_display_range=slice(-1, -num_display, -1),
                ignore_warnings=True,
            )

    def single_decision_plot(self, threshold, index, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            shap.decision_plot(
                threshold,
                self.shap_values[index],
                self.X_test,
                feature_names=self.feature_names,
                feature_display_range=slice(-1, -num_display, -1),
                ignore_warnings=True,
            )
