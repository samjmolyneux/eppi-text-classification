import warnings

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from . import shap_colors as colors

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

    def dot_plot(self, num_display=10, log_scale=True, plot_zero=False):
        if isinstance(self.model, LGBMClassifier):
            summary_new(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                max_display=num_display,
                color_bar_label="tfidf value",
                plot_type="dot",
                use_log_scale=log_scale,
                plot_size=(20, 0.4 * num_display),
                plot_zero=plot_zero,
            )

    def bar_chart(self, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            summary_new(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=num_display,
            )

    # def violin_plot(self, num_display=10):
    #     if isinstance(self.model, LGBMClassifier):
    #         shap.summary_plot(
    #             self.shap_values,
    #             self.X_test,
    #             feature_names=self.feature_names,
    #             plot_type="violin",
    #             max_display=num_display,
    #         )

    # TO DO: Double check that the use of threshold is right for the base value
    def decision_plot(self, threshold, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            decision(
                threshold,
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                feature_display_range=slice(-1, -num_display, -1),
                ignore_warnings=True,
            )

    def single_decision_plot(self, threshold, index, num_display=10):
        if isinstance(self.model, LGBMClassifier):
            decision(
                threshold,
                self.shap_values[index],
                self.X_test,
                feature_names=self.feature_names,
                feature_display_range=slice(-1, -num_display, -1),
                ignore_warnings=True,
            )


def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[: length_limit - 3] + "..."
    else:
        return text


def summary_new(
    shap_values,
    features=None,
    feature_names=None,
    max_display=None,
    plot_type=None,
    color=None,
    axis_color="#333333",
    title=None,
    alpha=1,
    show=True,
    sort=True,
    color_bar=True,
    plot_size="auto",
    layered_violin_max_num_bins=20,
    class_names=None,
    class_inds=None,
    color_bar_label=plot_labels["FEATURE_VALUE"],
    cmap=colors.red_blue,
    show_values_in_legend=False,
    # depreciated
    auto_size_plot=None,
    use_log_scale=False,
    plot_zero=False,
):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.

    show_values_in_legend: bool
        Flag to print the mean of the SHAP values in the multi-output bar plot. Set to False
        by default.

    """
    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    # deprecation warnings
    if auto_size_plot is not None:
        warnings.warn(
            "auto_size_plot=False is deprecated and is now ignored! Use plot_size=None instead. "
            "The parameter auto_size_plot will be removed in the next release 0.46.0.",
            DeprecationWarning,
        )

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar"  # default for multi-output explanations
        assert (
            plot_type == "bar"
        ), "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot"  # default for single output explanations
        assert (
            len(shap_values.shape) != 1
        ), "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == "layered_violin":
            color = "coolwarm"
        elif multi_class:

            def color(i):
                return colors.red_blue_circle(i / len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = shap_values[0].shape[1] if multi_class else shap_values.shape[1]

    if features is not None:
        shape_msg = (
            "The shape of the shap_values matrix does not match the shape of the "
            "provided data matrix."
        )
        if num_features - 1 == features.shape[1]:
            assert False, (
                shape_msg
                + " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? Of so just pass shap_values[:,:-1]."
            )
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array(
            [plot_labels["FEATURE"] % str(i) for i in range(num_features)]
        )

    # plotting SHAP interaction values
    if not multi_class and len(shap_values.shape) == 3:
        if plot_type == "compact_dot":
            new_shap_values = shap_values.reshape(shap_values.shape[0], -1)
            new_features = np.tile(features, (1, 1, features.shape[1])).reshape(
                features.shape[0], -1
            )

            new_feature_names = []
            for c1 in feature_names:
                for c2 in feature_names:
                    if c1 == c2:
                        new_feature_names.append(c1)
                    else:
                        new_feature_names.append(c1 + "* - " + c2)

            return summary_new(
                new_shap_values,
                new_features,
                new_feature_names,
                max_display=max_display,
                plot_type="dot",
                color=color,
                axis_color=axis_color,
                title=title,
                alpha=alpha,
                show=show,
                sort=sort,
                color_bar=color_bar,
                plot_size=plot_size,
                class_names=class_names,
                color_bar_label="*" + color_bar_label,
            )

        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        sort_inds = np.argsort(-np.abs(shap_values.sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (shap_values.shape[1] ** 2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        pl.subplot(1, max_display, 1)
        proj_shap_values = shap_values[:, sort_inds[0], sort_inds]
        proj_shap_values[:, 1:] *= 2  # because off diag effects are split in half
        summary_new(
            proj_shap_values,
            features[:, sort_inds] if features is not None else None,
            feature_names=feature_names[sort_inds],
            sort=False,
            show=False,
            color_bar=False,
            plot_size=None,
            max_display=max_display,
        )
        pl.xlim((slow, shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            pl.subplot(1, max_display, i + 1)
            proj_shap_values = shap_values[:, ind, sort_inds]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= (
                2  # because only off diag effects are split in half
            )
            summary_new(
                proj_shap_values,
                features[:, sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                plot_size=None,
                max_display=max_display,
            )
            pl.xlim((slow, shigh))
            pl.xlabel("")
            if i == min(len(sort_inds), max_display) // 2:
                pl.xlabel(plot_labels["INTERACTION_VALUE"])
            pl.title(shorten_text(feature_names[ind], title_length_limit))
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            pl.show()
        return

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(
                np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
            )
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    if use_log_scale:
        # TO DO: MAKE IT SO THAT WE ONLY INCLUDE THE VALUES THAT WILL BE PLOTTED
        temp_feats = shap_values[:, feature_order]
        min_non_zero_by_magnitude = np.min(np.abs(temp_feats[temp_feats != 0]))
        pl.xscale("symlog", linthresh=min_non_zero_by_magnitude, linscale=1)
        # pdb.set_trace()
        # pl.xscale('symlog')

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]:  # check categorical feature
                    colored_feature = False
                else:
                    values = np.array(
                        values, dtype=np.float64
                    )  # make sure this can be numeric
            except Exception:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(
                nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
            )
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
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

                assert features.shape[0] == len(
                    shaps
                ), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                pl.scatter(
                    shaps[nan_mask],
                    pos + ys[nan_mask],
                    color="#777777",
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    rasterized=len(shaps) > 500,
                )

                if plot_zero:
                    # plot the non-nan values colored by the trimmed feature value
                    cvals = values[np.invert(nan_mask)].astype(np.float64)
                    cvals_imp = cvals.copy()
                    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                    cvals[cvals_imp > vmax] = vmax
                    cvals[cvals_imp < vmin] = vmin
                    pl.scatter(
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
                        rasterized=len(shaps) > 500,
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
                    pl.scatter(
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
                        rasterized=len(shaps) > 500,
                    )

            else:
                pl.scatter(
                    shaps,
                    pos + ys,
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    color=color if colored_feature else "#777777",
                    rasterized=len(shaps) > 500,
                )

    elif not multi_class and plot_type == "bar":
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(shap_values).mean(0)
        pl.barh(
            y_pos, global_shap_values[feature_inds], 0.7, align="center", color=color
        )
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])

    elif multi_class and plot_type == "bar":
        if class_names is None:
            class_names = ["Class " + str(i) for i in range(len(shap_values))]
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        left_pos = np.zeros(len(feature_inds))

        if class_inds is None:
            class_inds = np.argsort(
                [-np.abs(shap_values[i]).mean() for i in range(len(shap_values))]
            )
        elif class_inds == "original":
            class_inds = range(len(shap_values))

        if show_values_in_legend:
            # Get the smallest decimal place of the first significant digit
            # to print on the legend. The legend will print ('n_decimal'+1)
            # decimal places.
            # Set to 1 if the smallest number is bigger than 1.
            smallest_shap = np.min(np.abs(shap_values).mean((1, 2)))
            if smallest_shap > 1:
                n_decimals = 1
            else:
                n_decimals = int(-np.floor(np.log10(smallest_shap)))

        for i, ind in enumerate(class_inds):
            global_shap_values = np.abs(shap_values[ind]).mean(0)
            if show_values_in_legend:
                label = f"{class_names[ind]} ({np.round(np.mean(global_shap_values),(n_decimals+1))})"
            else:
                label = class_names[ind]
            pl.barh(
                y_pos,
                global_shap_values[feature_inds],
                0.7,
                left=left_pos,
                align="center",
                color=color(i),
                label=label,
            )
            left_pos += global_shap_values[feature_inds]
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        pl.legend(frameon=False, fontsize=12)

    # draw the color bar
    if (
        color_bar
        and features is not None
        and plot_type != "bar"
        and (plot_type != "layered_violin" or color in pl.cm.datad)
    ):
        import matplotlib.cm as cm

        m = cm.ScalarMappable(
            cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color)
        )
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels(
            [plot_labels["FEATURE_VALUE_LOW"], plot_labels["FEATURE_VALUE_HIGH"]]
        )
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
    #         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    #         cb.ax.set_aspect((bbox.height - 0.9) * 20)
    # cb.draw_all()

    pl.gca().xaxis.set_ticks_position("bottom")
    pl.gca().yaxis.set_ticks_position("none")
    pl.gca().spines["right"].set_visible(False)
    pl.gca().spines["top"].set_visible(False)
    pl.gca().spines["left"].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(
        range(len(feature_order)),
        [feature_names[i] for i in feature_order],
        fontsize=13,
    )
    if plot_type != "bar":
        pl.gca().tick_params("y", length=20, width=0.5, which="major")
    pl.gca().tick_params("x", labelsize=11)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(plot_labels["GLOBAL_VALUE"], fontsize=13)
    else:
        pl.xlabel(plot_labels["VALUE"], fontsize=13)
    pl.tight_layout()

    # Delete the ticks closest to zero
    if plot_type == "dot" and use_log_scale:
        x_ticks = pl.gca().xaxis.get_major_ticks()

        for i, tick in enumerate(x_ticks):
            if (
                (tick.label1.get_text() == "$\mathdefault{0}$")
                and i > 0
                and i < len(x_ticks) - 1
            ):
                x_ticks[i - 1].label1.set_visible(False)
                x_ticks[i + 1].label1.set_visible(False)

    if show:
        pl.show()


from typing import Union

import matplotlib.cm as cm
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

# from ..utils import hclust_ordering
# from ..utils._legacy import LogitLink, convert_to_link
# from . import colors
# from ._labels import labels

import matplotlib.lines as mlines
import math


def __change_shap_base_value(base_value, new_base_value, shap_values) -> np.ndarray:
    """Shift SHAP base value to a new value. This function assumes that `base_value` and `new_base_value` are scalars
    and that `shap_values` is a two or three dimensional array.
    """
    # matrix of shap_values
    if shap_values.ndim == 2:
        return shap_values + (base_value - new_base_value) / shap_values.shape[1]

    # cube of shap_interaction_values
    main_effects = shap_values.shape[1]
    all_effects = main_effects * (main_effects + 1) // 2
    temp = (
        (base_value - new_base_value) / all_effects / 2
    )  # divided by 2 because interaction effects are halved
    shap_values = shap_values + temp
    # Add the other half to the main effects on the diagonal
    idx = np.diag_indices_from(shap_values[0])
    shap_values[:, idx[0], idx[1]] += temp
    return shap_values


def __decision_plot_matplotlib(
    base_value,
    cumsum,
    ascending,
    feature_display_count,
    features,
    feature_names,
    highlight,
    plot_color,
    axis_color,
    y_demarc_color,
    xlim,
    alpha,
    color_bar,
    auto_size_plot,
    title,
    show,
    legend_labels,
    legend_location,
):
    """Matplotlib rendering for decision_plot()"""
    # image size

    # TO DO: handle the edge case where there is no value that is different to the threshold
    row_height = 0.4
    if auto_size_plot:
        pl.gcf().set_size_inches(8, feature_display_count * row_height + 1.5)

    # draw horizontal dashed lines for each feature contribution
    for i in range(1, feature_display_count):
        pl.axhline(y=i, color=y_demarc_color, lw=0.5, dashes=(1, 5), zorder=-1)

    # initialize highlighting
    linestyle = np.array("-", dtype=object)
    linestyle = np.repeat(linestyle, cumsum.shape[0])
    linewidth = np.repeat(1, cumsum.shape[0])
    if highlight is not None:
        linestyle[highlight] = "-."
        linewidth[highlight] = 2

    # plot each observation's cumulative SHAP values.
    ax = pl.gca()
    ax.set_xlim(xlim)
    m = cm.ScalarMappable(cmap=plot_color)
    m.set_clim(xlim)
    y_pos = np.arange(0, feature_display_count + 1)
    lines = []
    for i in range(cumsum.shape[0]):
        o = pl.plot(
            cumsum[
                i, :
            ],  # -base_value is to make the plot centered around the threshold
            y_pos,
            color=m.to_rgba(cumsum[i, -1], alpha),
            linewidth=linewidth[i],
            linestyle=linestyle[i],
        )
        lines.append(o[0])

    # determine font size. if ' *\n' character sequence is found (as in interaction labels), use a smaller
    # font. we don't shrink the font for all interaction plots because if an interaction term is not
    # in the display window there is no need to shrink the font.
    s = next((s for s in feature_names if " *\n" in s), None)
    fontsize = 13 if s is None else 9

    # if there is a single observation and feature values are supplied, print them.
    if (cumsum.shape[0] == 1) and (features is not None):
        renderer = pl.gcf().canvas.get_renderer()
        inverter = pl.gca().transData.inverted()
        y_pos = y_pos + 0.5
        for i in range(feature_display_count):
            v = features[0, i]
            if isinstance(v, str):
                v = f"({str(v).strip()})"
            else:
                v = "({})".format(f"{v:,.3f}".rstrip("0").rstrip("."))
            t = ax.text(
                np.max(cumsum[0, i : (i + 2)]),
                y_pos[i],
                "  " + v,
                fontsize=fontsize,
                horizontalalignment="left",
                verticalalignment="center_baseline",
                color="#666666",
                zorder=1500,
            )
            bb = inverter.transform_bbox(t.get_window_extent(renderer=renderer))
            if bb.xmax > xlim[1]:
                t.set_text(v + "  ")
                t.set_x(np.min(cumsum[0, i : (i + 2)]))
                t.set_horizontalalignment("right")
                bb = inverter.transform_bbox(t.get_window_extent(renderer=renderer))
                if bb.xmin < xlim[0]:
                    t.set_text(v)
                    t.set_x(xlim[0])
                    t.set_horizontalalignment("left")

    # style axes
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(np.arange(feature_display_count) + 0.5, feature_names, fontsize=fontsize)
    ax.tick_params("x", labelsize=11)
    pl.ylim(0, feature_display_count)
    pl.xlabel("Model Output Value (Relative to Threshold)", fontsize=13)

    # draw the color bar - must come after axes styling
    if color_bar:
        m = cm.ScalarMappable(cmap=plot_color)
        m.set_array(np.array([0, 1]))

        # place the colorbar
        pl.ylim(0, feature_display_count + 0.25)
        ax_cb = ax.inset_axes(
            [xlim[0], feature_display_count, xlim[1] - xlim[0], 0.25],
            transform=ax.transData,
        )
        cb = pl.colorbar(m, ticks=[0, 1], orientation="horizontal", cax=ax_cb)
        cb.set_ticklabels([])
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(alpha)
        cb.outline.set_visible(False)

        # re-activate the main axis for drawing.
        pl.sca(ax)

    if title:
        # TODO decide on style/size
        pl.title(title)

    if ascending:
        pl.gca().invert_yaxis()

    if legend_labels is not None:
        ax.legend(handles=lines, labels=legend_labels, loc=legend_location)

    non_zero_mask = (cumsum) != 0
    min_non_zero = np.min(np.abs(cumsum[non_zero_mask]))
    pl.xscale("symlog", linthresh=min_non_zero, linscale=1)

    # Add details for threshold line
    pl.axvline(x=0, color="black", zorder=1000)
    threshold_line = mlines.Line2D([], [], color="black", label="Threshold")
    # text_y_position = ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04  # Slightly above the top
    text_y_position = ax.get_ylim()[1] + 0.4  # Slightly above the top

    ax.text(
        0,
        text_y_position,
        f"{base_value:.3} (Actual)",
        verticalalignment="top",
        horizontalalignment="left",
        zorder=1001,
    )
    ax.legend(handles=[threshold_line], loc="best")

    # Remove unwanted ticks
    pl.draw()
    x_ticks = ax.xaxis.get_major_ticks()
    for i, tick in enumerate(x_ticks):
        if (
            (tick.label1.get_text() == "$\mathdefault{0}$")
            and i > 0
            and i < len(x_ticks) - 1
        ):
            x_ticks[i - 1].label1.set_visible(False)
            x_ticks[i + 1].label1.set_visible(False)

    pl.xticks(rotation=45)

    if show:
        pl.show()


def decision(
    base_value,
    shap_values,
    features=None,
    feature_names=None,
    feature_order="importance",
    feature_display_range=None,
    highlight=None,
    link="identity",
    plot_color=None,
    axis_color="#333333",
    y_demarc_color="#333333",
    alpha=None,
    color_bar=True,
    auto_size_plot=True,
    title=None,
    xlim=None,
    show=True,
    return_objects=False,
    ignore_warnings=False,
    new_base_value=None,
    legend_labels=None,
    legend_location="best",
):
    """Visualize model decisions using cumulative SHAP values.

    Each plotted line explains a single model prediction. If a single prediction is plotted, feature values will be
    printed in the plot (if supplied). If multiple predictions are plotted together, feature values will not be printed.
    Plotting too many predictions together will make the plot unintelligible.

    Parameters
    ----------
    base_value : float or numpy.ndarray
        This is the reference value that the feature contributions start from. Usually, this is
        ``explainer.expected_value``.

    shap_values : numpy.ndarray
        Matrix of SHAP values (# features) or (# samples x # features) from
        ``explainer.shap_values()``. Or cube of SHAP interaction values (# samples x
        # features x # features) from ``explainer.shap_interaction_values()``.

    features : numpy.array or pandas.Series or pandas.DataFrame or numpy.ndarray or list
        Matrix of feature values (# features) or (# samples x # features). This provides the values of all the
        features and, optionally, the feature names.

    feature_names : list or numpy.ndarray
        List of feature names (# features). If ``None``, names may be derived from the
        ``features`` argument if a Pandas object is provided. Otherwise, numeric feature
        names will be generated.

    feature_order : str or None or list or numpy.ndarray
        Any of "importance" (the default), "hclust" (hierarchical clustering), ``None``,
        or a list/array of indices.

    feature_display_range: slice or range
        The slice or range of features to plot after ordering features by ``feature_order``. A step of 1 or ``None``
        will display the features in ascending order. A step of -1 will display the features in descending order. If
        ``feature_display_range=None``, ``slice(-1, -21, -1)`` is used (i.e. show the last 20 features in descending order).
        If ``shap_values`` contains interaction values, the number of features is automatically expanded to include all
        possible interactions: N(N + 1)/2 where N = ``shap_values.shape[1]``.

    highlight : Any
        Specify which observations to draw in a different line style. All numpy indexing methods are supported. For
        example, list of integer indices, or a bool array.

    link : str
        Use "identity" or "logit" to specify the transformation used for the x-axis. The "logit" link transforms
        log-odds into probabilities.

    plot_color : str or matplotlib.colors.ColorMap
        Color spectrum used to draw the plot lines. If ``str``, a registered matplotlib color name is assumed.

    axis_color : str or int
        Color used to draw plot axes.

    y_demarc_color : str or int
        Color used to draw feature demarcation lines on the y-axis.

    alpha : float
        Alpha blending value in [0, 1] used to draw plot lines.

    color_bar : bool
        Whether to draw the color bar (legend).

    auto_size_plot : bool
        Whether to automatically size the matplotlib plot to fit the number of features
        displayed. If ``False``, specify the plot size using matplotlib before calling
        this function.

    title : str
        Title of the plot.

    xlim: tuple[float, float]
        The extents of the x-axis (e.g. ``(-1.0, 1.0)``). If not specified, the limits
        are determined by the maximum/minimum predictions centered around base_value
        when ``link="identity"``. When ``link="logit"``, the x-axis extents are ``(0,
        1)`` centered at 0.5. ``xlim`` values are not transformed by the ``link``
        function. This argument is provided to simplify producing multiple plots on the
        same scale for comparison.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    return_objects : bool
        Whether to return a :obj:`DecisionPlotResult` object containing various plotting
        features. This can be used to generate multiple decision plots using the same
        feature ordering and scale.

    ignore_warnings : bool
        Plotting many data points or too many features at a time may be slow, or may create very large plots. Set
        this argument to ``True`` to override hard-coded limits that prevent plotting large amounts of data.

    new_base_value : float
        SHAP values are relative to a base value. By default, this base value is the
        expected value of the model's raw predictions. Use ``new_base_value`` to shift
        the base value to an arbitrary value (e.g. the cutoff point for a binary
        classification task).

    legend_labels : list of str
        List of legend labels. If ``None``, legend will not be shown.

    legend_location : str
        Legend location. Any of "best", "upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center".

    Returns
    -------
    DecisionPlotResult or None
        Returns a :obj:`DecisionPlotResult` object if ``return_objects=True``. Returns ``None`` otherwise (the default).
    """

    # code taken from force_plot. auto unwrap the base_value
    if type(base_value) == np.ndarray and len(base_value) == 1:
        base_value = base_value[0]

    if isinstance(base_value, list) or isinstance(shap_values, list):
        raise TypeError(
            "Looks like multi output. Try base_value[i] and shap_values[i], "
            "or use shap.multioutput_decision_plot()."
        )

    # validate shap_values
    if not isinstance(shap_values, np.ndarray):
        raise TypeError(
            "The shap_values arg is the wrong type. Try explainer.shap_values()."
        )

    # calculate the various dimensions involved (observations, features, interactions, display, etc.
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    observation_count = shap_values.shape[0]
    feature_count = shap_values.shape[1]

    # code taken from force_plot. convert features from other types.
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.to_list()
        features = features.values
    elif isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = features.index.to_list()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and features.ndim == 1 and feature_names is None:
        feature_names = features.tolist()
        features = None

    # the above code converts features to either None or np.ndarray. if features is something else at this point,
    # there's a problem.
    if not isinstance(features, (np.ndarray, type(None))):
        raise TypeError("The features arg uses an unsupported type.")
    if (features is not None) and (features.ndim == 1):
        features = features.reshape(1, -1)

    # validate/generate feature_names. at this point, feature_names does not include interactions.
    if feature_names is None:
        feature_names = [plot_labels["FEATURE"] % str(i) for i in range(feature_count)]
    elif len(feature_names) != feature_count:
        raise ValueError(
            "The feature_names arg must include all features represented in shap_values."
        )
    elif not isinstance(feature_names, (list, np.ndarray)):
        raise TypeError("The feature_names arg requires a list or numpy array.")

    # transform interactions cube to a matrix and generate interaction names.
    if shap_values.ndim == 3:
        # flatten
        triu_count = feature_count * (feature_count - 1) // 2
        idx_diag = np.diag_indices_from(shap_values[0])
        idx_triu = np.triu_indices_from(shap_values[0], 1)
        a = np.ndarray(
            (observation_count, feature_count + triu_count), shap_values.dtype
        )
        a[:, :feature_count] = shap_values[:, idx_diag[0], idx_diag[1]]
        a[:, feature_count:] = shap_values[:, idx_triu[0], idx_triu[1]] * 2
        shap_values = a
        # names
        a = [None] * shap_values.shape[1]
        a[:feature_count] = feature_names
        for i, row, col in zip(
            range(feature_count, shap_values.shape[1]), idx_triu[0], idx_triu[1]
        ):
            a[i] = f"{feature_names[row]} *\n{feature_names[col]}"
        feature_names = a
        feature_count = shap_values.shape[1]
        features = None  # Can't use feature values for interactions...

    # determine feature order
    if isinstance(feature_order, list):
        feature_idx = np.array(feature_order)
    elif isinstance(feature_order, np.ndarray):
        feature_idx = feature_order
    elif (feature_order is None) or (feature_order.lower() == "none"):
        feature_idx = np.arange(feature_count)
    elif feature_order == "importance":
        feature_idx = np.argsort(np.sum(np.abs(shap_values), axis=0))
    else:
        raise ValueError(
            "The feature_order arg requires 'importance', 'hclust', 'none', or an integer list/array "
            "of feature indices."
        )

    if (feature_idx.shape != (feature_count,)) or (
        not np.issubdtype(feature_idx.dtype, np.integer)
    ):
        raise ValueError(
            "A list or array has been specified for the feature_order arg. The length must match the "
            "feature count and the data type must be integer."
        )

    # validate and convert feature_display_range to a slice. prevents out of range errors later.
    if feature_display_range is None:
        feature_display_range = slice(
            -1, -21, -1
        )  # show last 20 features in descending order.
    elif not isinstance(feature_display_range, (slice, range)):
        raise TypeError("The feature_display_range arg requires a slice or a range.")
    elif feature_display_range.step not in (-1, 1, None):
        raise ValueError(
            "The feature_display_range arg supports a step of 1, -1, or None."
        )
    elif isinstance(feature_display_range, range):
        # Negative values in a range are not the same as negs in a slice. Consider range(2, -1, -1) == [2, 1, 0],
        # but slice(2, -1, -1) == [] when len(features) > 2. However, range(2, -1, -1) == slice(2, -inf, -1) after
        # clipping.
        a = np.iinfo(np.integer).min
        feature_display_range = slice(
            feature_display_range.start
            if feature_display_range.start >= 0
            else a,  # should never happen, but...
            feature_display_range.stop if feature_display_range.stop >= 0 else a,
            feature_display_range.step,
        )

    # apply new_base_value
    if new_base_value is not None:
        shap_values = __change_shap_base_value(base_value, new_base_value, shap_values)
        base_value = new_base_value

    # use feature_display_range to determine which features will be plotted. convert feature_display_range to
    # ascending indices and expand by one in the negative direction. why? we are plotting the change in prediction
    # for every feature. this requires that we include the value previous to the first displayed feature
    # (i.e. i_0 - 1 to i_n).
    a = feature_display_range.indices(feature_count)
    ascending = True
    if a[2] == -1:  # The step
        ascending = False
        a = (a[1] + 1, a[0] + 1, 1)
    feature_display_count = a[1] - a[0]
    shap_values = shap_values[:, feature_idx]
    if a[0] == 0:
        cumsum = np.ndarray(
            (observation_count, feature_display_count + 1), shap_values.dtype
        )
        cumsum[:, 0] = 0
        cumsum[:, 1:] = np.nancumsum(shap_values[:, 0 : a[1]], axis=1)
    else:
        cumsum = np.nancumsum(shap_values, axis=1)[:, (a[0] - 1) : a[1]]

    # Select and sort feature names and features according to the range selected above
    feature_names = np.array(feature_names)
    feature_names_display = feature_names[feature_idx[a[0] : a[1]]].tolist()
    feature_names = feature_names[feature_idx].tolist()
    features_display = (
        None if features is None else features[:, feature_idx[a[0] : a[1]]]
    )

    # throw large data errors
    if not ignore_warnings:
        if observation_count > 2000:
            raise RuntimeError(
                f"Plotting {observation_count} observations may be slow. Consider subsampling or set "
                "ignore_warnings=True to ignore this message."
            )
        if feature_display_count > 200:
            raise RuntimeError(
                f"Plotting {feature_display_count} features may create a very large plot. Set "
                "ignore_warnings=True to ignore this "
                "message."
            )
        if feature_count * observation_count > 100000000:
            raise RuntimeError(
                f"Processing SHAP values for {feature_count} features over {observation_count} observations may be slow. Set "
                "ignore_warnings=True to ignore this "
                "message."
            )

    # convert values based on link and update x-axis extents

    # TO DO: This whole section needs changing
    create_xlim = xlim is None
    base_value_saved = base_value

    if create_xlim:
        xmin = np.min((cumsum.min(), 0))
        xmax = np.max((cumsum.max(), 0))
        # create a symmetric axis around base_value
        if abs(xmin) > xmax:
            xlim = (xmin, -xmin)
        else:
            xlim = (-xmax, xmax)
        # Adjust xlim to include a little visual margin.
        a = (xlim[1] - xlim[0]) * 0.02
        xlim = (xlim[0] - a, xlim[1] + a)

    # Initialize style arguments
    if alpha is None:
        alpha = 1.0

    if plot_color is None:
        plot_color = colors.red_blue

    __decision_plot_matplotlib(
        base_value,
        cumsum,
        ascending,
        feature_display_count,
        features_display,
        feature_names_display,
        highlight,
        plot_color,
        axis_color,
        y_demarc_color,
        xlim,
        alpha,
        color_bar,
        auto_size_plot,
        title,
        show,
        legend_labels,
        legend_location,
    )


# def multioutput_decision(
#     base_values, shap_values, row_index, **kwargs
# ) -> Union[DecisionPlotResult, None]:
#     """Decision plot for multioutput models.

#     Plots all outputs for a single observation. By default, the plotted base value will be the mean of base_values
#     unless new_base_value is specified. Supports both SHAP values and SHAP interaction values.

#     Parameters
#     ----------
#     base_values : list of float
#         This is the reference value that the feature contributions start from. Use explainer.expected_value.

#     shap_values : list of numpy.ndarray
#         A multioutput list of SHAP matrices or SHAP cubes from explainer.shap_values() or
#         explainer.shap_interaction_values(), respectively.

#     row_index : int
#         The integer index of the row to plot.

#     **kwargs : Any
#         Arguments to be passed on to decision_plot().

#     Returns
#     -------
#     DecisionPlotResult or None
#         Returns a DecisionPlotResult object if `return_objects=True`. Returns `None` otherwise (the default).

#     """
#     if not (isinstance(base_values, list) and isinstance(shap_values, list)):
#         raise ValueError("The base_values and shap_values args expect lists.")

#     # convert arguments to arrays for simpler handling
#     base_values = np.array(base_values)
#     if not ((base_values.ndim == 1) or (np.issubdtype(base_values.dtype, np.number))):
#         raise ValueError("The base_values arg should be a list of scalars.")
#     shap_values = np.array(shap_values)
#     if shap_values.ndim not in [3, 4]:
#         raise ValueError(
#             "The shap_values arg should be a list of two or three dimensional SHAP arrays."
#         )
#     if shap_values.shape[0] != base_values.shape[0]:
#         raise ValueError("The base_values output length is different than shap_values.")

#     # shift shap base values to mean of base values
#     base_values_mean = base_values.mean()
#     for i in range(shap_values.shape[0]):
#         shap_values[i] = __change_shap_base_value(
#             base_values[i], base_values_mean, shap_values[i]
#         )

#     # select the feature row corresponding to row_index
#     if (kwargs is not None) and ("features" in kwargs):
#         features = kwargs["features"]
#         if isinstance(features, np.ndarray) and (features.ndim == 2):
#             kwargs["features"] = features[[row_index]]
#         elif isinstance(features, pd.DataFrame):
#             kwargs["features"] = features.iloc[row_index]

#     return decision(base_values_mean, shap_values[:, row_index, :], **kwargs)
