from shap import TreeExplainer


def my_func(model, array, y_test):
    explainer = TreeExplainer(
        model, array, feature_perturbation="interventional", model_output="raw"
    )
    tree_values_all = explainer.shap_values(array, y_test)
