"""For plotting the SHAP values of a model's features."""

import warnings
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.text import Text
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from . import predict_scores
from . import shap_colors as colors

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier


# Considerations: The number of samples that are calculated to get the kernel explainer
# shap values should be adjusted

# TO DO: REDO Doumentation
# TO DO: Make the Plot objects figure specific

plot_labels = {
    "MAIN_EFFECT": "SHAP main effect value for\n%s",
    "INTERACTION_VALUE": "SHAP interaction value",
    "INTERACTION_EFFECT": "SHAP interaction value for\n%s and %s",
    "VALUE": "SHAP value (impact on model output)",
    "GLOBAL_VALUE": "mean(|SHAP value|) (average impact on model output magnitude)",
    "VALUE_FOR": "SHAP value for\n%s",
    "PLOT_FOR": "SHAP plot for %s",
    "FEATURE": "Feature %s",
    "FEATURE_VALUE": "Feature value",
    "FEATURE_VALUE_LOW": "Low",
    "FEATURE_VALUE_HIGH": "High",
    "JOINT_VALUE": "Joint SHAP value",
    "MODEL_OUTPUT": "Model output value",
}


# TO DO, do the same for the other plots
class DecisionPlot:
    """
    Class for managing decision plot of SHAP values.

    This object should be accessed through the ShapPlotter class.
    """

    def __init__(
        self,
        expected_value: float,
        threshold: float,
        shap_values: csr_matrix,
        X_test: csr_matrix,
        feature_names: NDArray[np.str_],
        num_display: int,
        log_scale: bool,
    ) -> None:
        """
        Set up the DecisionPlot object.

        Parameters
        ----------
        expected_value : float
            Expect value from the SHAP explainer.

        threshold : float
            Decision threshold for the model output.

        shap_values : csr_matrix
            SHAP values for the model's features. (#Samples, #Features)

        X_test : csr_matrix
            Data used for calculating SHAP values. (#Samples, #Features)

        feature_names : list[str]
            List of feature names corresponding to columns in X_test. (#Features,)

        num_display : int
            Number of features to display in the plot, from most important to least.

        log_scale : bool
            Wether to use a log scale for the x-axis.

        """
        self.expected_value = expected_value
        self.threshold = threshold
        self.shap_values = shap_values
        self.X_test = X_test
        self.feature_names = feature_names
        self.num_display = num_display
        self.log_scale = log_scale

    def show(self) -> None:
        """Display the decision plot."""
        self._make_plot()
        plt.show(block=False)

    def save(self, filename: str) -> None:
        """Save the decision plot."""
        self._make_plot()
        plt.savefig(filename)

    def _make_plot(self) -> None:
        # Create a decision plot
        X_test = self.X_test.toarray()
        shap_values = self.shap_values.toarray()

        # Create decision plot, ignoring unproblematic warnings from the shap package
        with warnings.catch_warnings():
            ignore_unproblematic_warnings_from_decision_plot()
            shap.decision_plot(
                base_value=self.expected_value,
                shap_values=shap_values,
                features=X_test,
                feature_names=self.feature_names,
                feature_display_range=slice(-1, -self.num_display, -1),
                ignore_warnings=True,
                show=False,
            )

        # Adjust for our use case
        ax = plt.gca()
        if self.log_scale:
            self._legacy_log_decision_changes(ax, self.threshold)
        else:
            self._legacy_decision_changes(ax, self.threshold)

        # Handle case that all shap values are zero
        num_of_non_zero_shap_values = np.count_nonzero(shap_values)
        if num_of_non_zero_shap_values == 0:
            add_warning_to_plot_that_all_shap_values_are_zero(ax)

    def _legacy_decision_changes(self, ax: Axes, threshold: float) -> None:
        """
        Domain specific changes to the decision plot for our use case.

        The function adds a threshold line, adjusts the x-axis limits, and
        makes visual improvements to the plot.

        Parameters
        ----------
        ax : A plt.axes.Axes object
            A legacy decision plot to make changes to.
            Specific changes for our application.

        threshold : float
            Decision threshold for the model output.
            Used to add a threshold line to the plot.

        """
        for line in ax.lines:
            if line.get_color() == "#999999":
                line.set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")

        add_new_threshold_line_to_decision_plot(ax, threshold)

        # Adjust x-axis to include the new threshold line on the plot.
        x_lims = ax.get_xlim()
        if threshold < x_lims[0]:
            ax.set_xlim(left=(threshold - 0.1 * (x_lims[1] - x_lims[0])))
        elif threshold > x_lims[1]:
            ax.set_xlim(right=(threshold + 0.1 * (x_lims[1] - x_lims[0])))

    # TO DO: Make a github issue for making these graphs dynamic with user input
    def _legacy_log_decision_changes(self, ax: Axes, threshold: float) -> None:
        """
        Domain specific changes to the log decision plot for our use case.

        The function adds a threshold line, adjusts the x-axis limits, and
        makes visual improvements to the plot.

        Parameters
        ----------
        ax : A plt.axes.Axes object
            A legacy decision plot to make changes to.
            Specific changes for our application.

        threshold : float
            Decision threshold for the model output.
            Used to add a threshold line to the plot.

        """
        for line in ax.lines:
            # Remove old threshold
            if line.get_color() == "#999999":
                line.set_visible(False)

            # Use transformation to center the plot lines around x=0
            elif line.get_color() != "#333333":
                x_data = np.array(line.get_xdata(), dtype=np.float64)
                line.set_xdata(x_data - threshold)

        # Remove the old colour bar
        inset_axes_list = [
            child for child in ax.get_children() if isinstance(child, Axes)
        ]
        inset_axes_list[0].set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")

        # Add new threshold line
        add_new_threshold_line_to_decision_plot(ax, 0)

        # Transform x-axis to include the threshold line on the plot
        x_lims = ax.get_xlim()
        ax.set_xlim(left=min(x_lims[0] - threshold, 0))
        ax.set_xlim(right=max(x_lims[1] - threshold, 0))

        plot_lines = [
            line
            for line in ax.lines
            if line.get_color() != "#333333"  # Horizontal dashed lines
            and line.get_color() != "#999999"  # Old threshold line
            and line.get_color() != "black"  # New threshold line
        ]

        # Transform x-axis to log scale if there are non-zero values
        x_data = np.array([], dtype=np.float64)
        for line in plot_lines:
            x_data = np.append(x_data, line.get_xdata())
        if np.count_nonzero(x_data) > 0:
            non_zero_xmin = np.min(np.abs(x_data[x_data != 0]))
            plt.xscale("symlog", linthresh=non_zero_xmin, linscale=1)

        # Removes the old text if there is a single observation.
        # TO DO: TRANSFORM THE TEXT.
        if len(plot_lines) == 1:
            text_elements = [
                child for child in ax.get_children() if isinstance(child, Text)
            ]
            for text in text_elements:
                text.set_visible(False)

        # Remove x tick labels near to origin to prevent clutter
        remove_tick_labels_adjacent_to_origin(ax)

        ax.set_xlabel("Model output value relative to threshold")


class DotPlot:
    """Class for managing dot plot of SHAP values."""

    def __init__(
        self,
        shap_values: csr_matrix,
        X_test: csr_matrix,
        feature_names: NDArray[np.str_],
        num_display: int,
        log_scale: bool,
        plot_zero: bool,
    ) -> None:
        """
        Initialize the DotPlot object.

        shap_values : csr_matrix
            SHAP values for the model's features. (#Samples, #Features).

        X_test : csr_matrix
            Data used for calculating SHAP values. (#Samples, #Features)

        feature_names : list[str]
            List of feature names corresponding to columns in X_test. (#Features,)

        num_display : int
            Number of features to display in the plot, from most important to least.

        log_scale : bool
            Wether to use a log scale for the x-axis.

        plot_zero : bool
            Wether to plot the SHAP values equal to zero.

        """
        self.shap_values = shap_values
        self.X_test = X_test
        self.feature_names = feature_names
        self.num_display = num_display
        self.log_scale = log_scale
        self.plot_zero = plot_zero

    def show(self) -> None:
        """Display the dot plot."""
        self._make_plot()
        plt.show(block=False)

    def save(self, filename: str) -> None:
        """Save the dot plot."""
        self._make_plot()
        plt.savefig(filename)

    def _make_plot(self) -> None:
        summary_new(
            shap_values=self.shap_values,
            features=self.X_test,
            feature_names=self.feature_names,
            max_display=self.num_display,
            color_bar_label="tfidf value",
            plot_type="dot",
            use_log_scale=self.log_scale,
            plot_size=(20, 0.4 * self.num_display),
            plot_zero=self.plot_zero,
        )


class BarPlot:
    """Class for managing bar plot of SHAP values."""

    def __init__(
        self,
        shap_values: csr_matrix,
        X_test: csr_matrix,
        feature_names: NDArray[np.str_],
        num_display: int,
    ) -> None:
        """
        Initialize the Barplot object.

        shap_values : csr_matrix
            SHAP values for the model's features. (#Samples, #Features).

        X_test : csr_matrix
            Data used for calculating SHAP values. (#Samples, #Features)

        feature_names : list[str]
            List of feature names corresponding to columns in X_test. (#Features,)

        num_display : int
            Number of features to display in the plot, from most important to least.

        """
        self.shap_values = shap_values
        self.X_test = X_test
        self.feature_names = feature_names
        self.num_display = num_display

    def show(self) -> None:
        """Display the bar plot."""
        self._make_plot()
        plt.show(block=False)

    def save(self, filename: str) -> None:
        """Save the bar plot."""
        self._make_plot()
        plt.savefig(filename)

    def _make_plot(self) -> None:
        summary_new(
            shap_values=self.shap_values,
            features=self.X_test,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=self.num_display,
        )


class ShapPlotter:
    """
    Plot graphs of SHAP values for a model's features.

    For each model type, the SHAP values are calculated differently.
    To add new models, you must setup functions for the model type.
    See https://shap.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        model: "LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC",
        X_test: csr_matrix,
        feature_names: NDArray[np.str_],
        tree_path_dependent: bool = False,
        kernel_nsamples: int | str = "auto",
        tree_explainer_working: bool = False,
    ) -> None:
        """
        Initialize the ShapPlotter object.

        There are two potential explainers to use, the tree explainer and the kernel.
        The tree explainer is currently broken and so the tree_explainer_working flag
        should be set to false. Also, the tree_path_dependent flag has no affect when
        the tree explainer is broken.

        The sparsity for the kernel explainer is also currently not working as intended.
        I will be using a pull request to change the package. In the mean time, X_test
        will be limited to 5GB of memory.

        Parameters
        ----------
        model : LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC
            Model to calculate SHAP values.

        X_test : csr_matrix
            Data to use on model for calculating SHAP values. (#Samples, #Features)

        feature_names : list[str]
            List of feature names corresponding to columns in X_test. (#Features,)

        tree_path_dependent : bool, optional
            This varible does nothing if the tree_explainer is not working.
            When using the tree explainer, this variable determines wether to calculate
            tree-path dependent SHAP values (Ignore X_test).
            By default False

        kernel_nsamples : int | str, optional
            This variable only has an affect when using the kernel explainer.
            The number of samples to use for approximating the SHAP values when using
            the kernel explainer. When auto, nsamples = 2 * X.shape[1] + 2048.
            By default "auto".

        tree_explainer_working : bool, optional
            Temporary flag to check if the tree explainer is working or not.
            Should be deleted once the tree explainer is fixed for sparse matrices.

        """
        check_size_xtest_not_too_large(X_test)
        self.model = model
        self.X_test = X_test
        self.feature_names = feature_names
        self.tree_path_dependent = tree_path_dependent
        self.kernel_nsamples = kernel_nsamples

        self.background_data = csr_matrix((1, X_test.shape[-1]))

        self.shap_values: csr_matrix
        self.expected_value: float

        # Setup the explantation for plotting
        if tree_explainer_working:
            model_name = self.model.__class__.__name__
            perform_explanation_func = getattr(
                self, f"{model_name.lower()}_setup", None
            )
        else:
            perform_explanation_func = self.temporary_setup_for_kernel

        if perform_explanation_func is None:
            msg = f"No setup function for model type {model_name}"
            raise ValueError(msg)
        perform_explanation_func()

    def dot_plot(
        self, num_display: int = 10, log_scale: bool = True, plot_zero: bool = False
    ) -> DotPlot:
        """
        Create a dot plot of the SHAP values.

        Parameters
        ----------
        num_display : int, optional
            Number of features to display in the plot, from most important to least.
            By default 10

        log_scale : bool, optional
            Wether to use a log scale for the x-axis. By default True

        plot_zero : bool, optional
            Wether to plot the SHAP values equal to zero.
            Due to the sparsity of the data, this is reccomended to be set to False.
            By default False

        """
        return DotPlot(
            self.shap_values,
            self.X_test,
            self.feature_names,
            num_display,
            log_scale,
            plot_zero,
        )

    def bar_chart(self, num_display: int = 10) -> BarPlot:
        """
        Plot the SHAP values in a bar chart.

        Parameters
        ----------
        num_display : int, optional
            Number of most important features to display in the plot,
            from most important to least.

        """
        return BarPlot(
            self.shap_values,
            self.X_test,
            feature_names=self.feature_names,
            num_display=num_display,
        )

    def decision_plot(
        self, threshold: float, num_display: int = 10, log_scale: bool = False
    ) -> DecisionPlot:
        """
        Plot the SHAP values in a decision plot.

        Parameters
        ----------
        threshold : float
            Model output threshold to use for the decision plot.
            This should be the raw output threshold used to classify the data.

        num_display : int, optional
            Number of most important features to display in the plot,
            from most important to least, by default 10

        log_scale : bool, optional
            Wether to use a log scale for the x-axis, by default False

        """
        return DecisionPlot(
            self.expected_value,
            threshold,
            self.shap_values,
            self.X_test,
            self.feature_names,
            num_display,
            log_scale,
        )

    def single_decision_plot(
        self,
        threshold: float,
        index: int,
        num_display: int = 10,
        log_scale: bool = False,
    ) -> DecisionPlot:
        """
        Plot the SHAP values in a decision plot for a single sample.

        Parameters
        ----------
        threshold : float
            Model output threshold to use for the decision plot.
            This should be the raw output threshold used to classify the data.

        index : int
            X_test index of the sample to plot.

        num_display : int, optional
            Number of most important features to display in the plot,
            from most important to least, by default 10

        log_scale : bool, optional
            Wether to use a log scale for the x-axis, by default False

        """
        return DecisionPlot(
            self.expected_value,
            threshold,
            self.shap_values[index],
            self.X_test[index],
            self.feature_names,
            num_display,
            log_scale,
        )

    def lgbmclassifier_setup(self) -> None:
        """Set up ShapPlotter for lgbm models."""
        self.setup_tree_explainer()
        self.shap_values = self.explainer.shap_values(self.X_test)
        self.expected_value = self.explainer.expected_value

    def xgbclassifier_setup(self) -> None:
        """Set up ShapPlotter for xgb models."""
        self.setup_tree_explainer()
        self.shap_values = self.explainer.shap_values(self.X_test)
        self.expected_value = self.explainer.expected_value

    def randomforestclassifier_setup(self) -> None:
        """Set up ShapPlotter for random forest models."""
        self.setup_tree_explainer()
        self.shap_values = self.explainer.shap_values(self.X_test)
        self.shap_values = self.shap_values[:, :, 1]
        self.expected_value = self.explainer.expected_value[1]

    def svc_setup(self) -> None:
        """Set up ShapPlotter for SVC models."""
        self.explainer = shap.KernelExplainer(
            self.model.decision_function, self.background_data
        )
        self.shap_values = self.explainer.shap_values(
            self.X_test, nsamples=self.kernel_nsamples, l1_reg="aic"
        )
        self.expected_value = self.explainer.expected_value

    def temporary_setup_for_kernel(self) -> None:
        """
        Set up ShapPlotter for models with kernel explainer.

        This is only to be used temporarily until the treeExplainer is fixed for sparse
        matrices.
        """
        # Set up the explainer object
        self.explainer = shap.KernelExplainer(
            self.temporary_prediction_wrapper_for_kernel_explainer, self.background_data
        )

        # Run the explaination and ignore the warnings
        with warnings.catch_warnings():
            ignore_unproblematic_warnings_from_kernel_explainer()

            shap_values = self.explainer.shap_values(
                self.X_test, nsamples=self.kernel_nsamples, l1_reg="aic", silent=True
            )
        self.shap_values = csr_matrix(shap_values)
        self.expected_value = self.explainer.expected_value

    def setup_tree_explainer(self) -> None:
        """General method to set up tree explainer for tree-based models."""
        if not self.tree_path_dependent:
            self.explainer = shap.TreeExplainer(
                self.model,
                data=self.background_data,
                model_output="raw",
                feature_perturbation="interventional",
            )
        else:
            self.explainer = shap.TreeExplainer(
                self.model,
                model_output="raw",
                feature_perturbation="tree_path_dependent",
            )

    def temporary_prediction_wrapper_for_kernel_explainer(
        self, X: csr_matrix
    ) -> NDArray[np.float64] | NDArray[np.float32]:
        """
        Temporary function to make predictions for the kernel explainer.

        This function is only to be used temporarily until the treeExplainer is fixed
        for sparse matrices.

        Parameters
        ----------
        X : csr_matrix
            Data to make predictions on.

        Returns
        -------
        NDArray[np.float64]
            Predictions for X.

        """
        scores = predict_scores(self.model, X)
        return scores


def check_size_xtest_not_too_large(dataset: csr_matrix) -> None:
    """Check the size of the dataset for the SHAP plotter."""
    if len(dataset.shape) != 2:
        msg = f"Dataset must have two dimensions, but got {dataset.shape}."
        raise ValueError(msg)
    if dataset.shape[0] > 2500:
        msg = f"""We have temporarily limited the size of the dataset to 2500 samples.
        You have used {dataset.shape[0]} samples."""
        raise MemoryError(msg)


# TO DO: separate into two functions for dot and bars
def summary_new(
    shap_values: csr_matrix,
    features: csr_matrix,
    feature_names: NDArray[np.str_],
    plot_type: str,
    max_display: int = 10,
    alpha: float = 1,
    color_bar: bool = True,
    plot_size: tuple[float, float] | None = None,
    color_bar_label: str = plot_labels["FEATURE_VALUE"],
    cmap: "LinearSegmentedColormap" = colors.red_blue,
    use_log_scale: bool = True,
    plot_zero: bool = False,
    n_y_bins: float = 100,
) -> None:
    """
    Plot the SHAP values in a dot plot or bar chart.

    This method is a rewritten version of the shap.summary_legacy() method.
    This method should be called from the ShapPlotter class,
    not directly. It can be called using bar_chart() or dot_plot().
    It is reccomended to use the log_scale option for the dot plot.

    Parameters
    ----------
    shap_values : csr_matrix
        SHAP values for the model's features. (#Samples, #Features)

    features : csr_matrix
        Data to use on model for calculating SHAP values. (#Samples, #Features)

    feature_names : NDArray[np.str_]
        List of feature names corresponding to columns in X_test. (#Features,)

    plot_type : str
        "dot" or "bar" to choose the type of plot to display, by default None

    max_display : int, optional
        Number of features to display in the plot, from most important to least.
        By default, 10.

    alpha : float, optional
        A measure of the opacity of the data points in [0, 1].
        By default 1.

    color_bar : bool, optional
        Whether to display the color bar.
        By default True.

    plot_size : int | tuple | str, optional
        The size of the plot.
        By default "auto"

    color_bar_label : str, optional
        Label for the color bar. Should be what the unit of measurement for the
        features. E.g. tfidf scores.
        By default plot_labels["FEATURE_VALUE"]

    cmap : LinearSegmentedColormap, optional
        A colourmap for displaying the points.
        By default colors.red_blue.

    use_log_scale : bool, optional
        Use a logarithm scale along the x-axis.
        By default False

    plot_zero : bool, optional
        Whether to (datapoint, feature) pairs that have a shap value equal to 0.
        It is reccomended setting this to false when working with sparse data,
        as is the case with tfidf vectorizer.
        By default False.

    n_y_bins : int, optional
        Parameter that controls the spread of points vertically within a feature.

    """
    shap_values = shap_values.toarray()
    features = features.toarray()

    num_of_non_zero_shap_values = np.count_nonzero(shap_values)

    color = colors.blue_rgb

    rasterize_threshold = 500

    # order features by the sum of their effect magnitudes
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)) :]

    if use_log_scale and num_of_non_zero_shap_values > 0:
        temp_feats = shap_values[:, feature_order]
        min_non_zero_by_magnitude = np.min(np.abs(temp_feats[temp_feats != 0]))
        plt.xscale("symlog", linthresh=min_non_zero_by_magnitude, linscale=1)

    row_height = 0.4
    if plot_size is None:
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    else:
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    plt.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = features[:, i]
            inds = np.arange(len(shaps))
            np.random.default_rng().shuffle(inds)
            values = values[inds]
            shaps = shaps[inds]

            colored_feature = True
            values = np.array(values, dtype=np.float64)

            # Add y bins to prevent point overlap for two points with same shap value
            num_points = len(shaps)
            quant = np.round(
                n_y_bins
                * (shaps - np.min(shaps))
                / (np.max(shaps) - np.min(shaps) + 1e-8)
            )
            inds = np.argsort(
                quant + np.random.default_rng().standard_normal(num_points) * 1e-6
            )
            layer = 0
            last_bin = -1
            ys = np.zeros(num_points)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax:  # fixes rare numerical precision issues
                    vmin = vmax

                if features.shape[0] != len(shaps):
                    msg = "Feature and SHAP matrices must have the same number of rows!"
                    raise ValueError(msg)

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                plt.scatter(
                    shaps[nan_mask],
                    pos + ys[nan_mask],
                    color="#777777",
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    rasterized=len(shaps) > rasterize_threshold,
                )

                if plot_zero:
                    # plot the non-nan values colored by the trimmed feature value
                    cvals = values[np.invert(nan_mask)].astype(np.float64)
                    cvals_imp = cvals.copy()
                    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                    cvals[cvals_imp > vmax] = vmax
                    cvals[cvals_imp < vmin] = vmin
                    plt.scatter(
                        shaps[np.invert(nan_mask)],
                        pos + ys[np.invert(nan_mask)],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=16,
                        c=cvals,
                        alpha=alpha,
                        linewidth=0,
                        zorder=3,
                        rasterized=len(shaps) > rasterize_threshold,
                    )

                else:
                    non_zero_mask = shaps != 0

                    # plot the non-nan values colored by the trimmed feature value
                    cvals = values[np.invert(nan_mask) & non_zero_mask].astype(
                        np.float64
                    )
                    cvals_imp = cvals.copy()
                    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                    cvals[cvals_imp > vmax] = vmax
                    cvals[cvals_imp < vmin] = vmin
                    plt.scatter(
                        shaps[np.invert(nan_mask) & non_zero_mask],
                        pos + ys[np.invert(nan_mask) & non_zero_mask],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=16,
                        c=cvals,
                        alpha=alpha,
                        linewidth=0,
                        zorder=3,
                        rasterized=len(shaps) > rasterize_threshold,
                    )

            else:
                plt.scatter(
                    shaps,
                    pos + ys,
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    color=color if colored_feature else "#777777",
                    rasterized=len(shaps) > rasterize_threshold,
                )

    elif plot_type == "bar":
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(shap_values).mean(0)
        plt.barh(
            y_pos, global_shap_values[feature_inds], 0.7, align="center", color=color
        )
        plt.yticks(y_pos, fontsize=13)
        plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])

    if color_bar and plot_type != "bar":
        m = cm.ScalarMappable(cmap=cmap)
        m.set_array([0, 1])
        cb = plt.colorbar(m, ax=plt.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels(
            [plot_labels["FEATURE_VALUE_LOW"], plot_labels["FEATURE_VALUE_HIGH"]]
        )
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        for spine in cb.ax.spines.values():
            spine.set_visible(False)

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().tick_params(color="#333333", labelcolor="#333333")
    plt.yticks(
        range(len(feature_order)),
        [feature_names[i] for i in feature_order],
        fontsize=13,
    )
    if plot_type != "bar":
        plt.gca().tick_params("y", length=20, width=0.5, which="major")
    plt.gca().tick_params("x", labelsize=11)
    plt.ylim(-1, len(feature_order))
    if plot_type == "bar":
        plt.xlabel(plot_labels["GLOBAL_VALUE"], fontsize=13)
    else:
        plt.xlabel(plot_labels["VALUE"], fontsize=13)
    plt.tight_layout()

    if num_of_non_zero_shap_values == 0:
        add_warning_to_plot_that_all_shap_values_are_zero(plt.gca())

    # Delete the ticks closest to zero if using log scale
    if plot_type == "dot" and use_log_scale:
        remove_tick_labels_adjacent_to_origin(plt.gca())


def add_warning_to_plot_that_all_shap_values_are_zero(ax: Axes) -> None:
    """Add a warning to the plot that all shap values are zero."""
    warning_message = """All SHAP values are zero. This means removing words from
     this data does not affect the model output."""
    ax.set_title(
        warning_message,
        color="red",
        fontsize=12,
        fontweight="bold",
    )


def remove_tick_labels_adjacent_to_origin(ax: Axes) -> None:
    """Remove the x-axis tick labels closest to origin to prevent overlap."""
    x_ticks = ax.xaxis.get_major_ticks()
    x_ticks_locs = ax.xaxis.get_ticklocs()
    for i, loc in enumerate(x_ticks_locs):
        if (loc == 0) and i > 0 and i < len(x_ticks) - 1:
            x_ticks[i - 1].label1.set_visible(False)
            x_ticks[i + 1].label1.set_visible(False)


def add_new_threshold_line_to_decision_plot(ax: Axes, threshold: float) -> None:
    """Add a new threshold line to the decision plot."""
    ax.axvline(x=threshold, color="black", zorder=1000)
    threshold_line = mlines.Line2D([], [], color="black", label="Threshold")
    ax.legend(handles=[threshold_line], loc="best")
    text_y_position = ax.get_ylim()[1] + 0.4  # Slightly above the top
    ax.text(
        threshold,
        text_y_position,
        f"{threshold:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        zorder=1001,
    )


def ignore_unproblematic_warnings_from_kernel_explainer() -> None:
    """
    Ignore warnings from kernel explainer that do not affect the output.

    Each warning is thrown due to some problem within the kernel explainer that has not
    been fixed. To remove these warnings, we require a pull request to the shap package.
    """
    # LGBM throws this warning when kernel explainer tries to predict with a lil_matrix
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Converting data to scipy sparse matrix.",
    )
    # Xgbost throws this warning when kernel explainer predicts with a lil_matrix
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=(
            "Unknown data type: <class 'scipy.sparse._lil.lil_matrix'>, "
            "trying to convert it to csr_matrix"
        ),
    )
    # Numpy throws this when kernel explainer tries to run a regression on uniform data
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="divide by zero encountered in log",
    )
    # Numpy throws this when kernel explainer tries to run a regression on uniform data
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in divide",
    )


def ignore_unproblematic_warnings_from_decision_plot() -> None:
    """Ignore warnings from decision plot that do not affect the output."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=(
            "Attempting to set identical low and high xlims makes transformation "
            "singular; automatically expanding."
        ),
    )
