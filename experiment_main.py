"""This module contains the main experiment function."""
import time
from copy import copy
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from shap.plots import force, waterfall
from shap import Explanation
from scipy.special import logit, expit
from tqdm import tqdm
from shap import TreeExplainer

from plotting.network import draw_interaction_network
from plotting.plot_n_sii import transform_interactions_in_n_shapley, plot_n_sii
from tree_shap_iq.conversion import convert_tree_estimator, TreeModel
from tree_shap_iq import TreeShapIQ
from tree_shap_iq.utils import powerset


def run_shap_iq(
        tree_model: Union[list[TreeModel], TreeModel],
        x_explain: np.ndarray,
        max_interaction_order: int,
        n_features: int,
        background_dataset: np.ndarray,
        observational: bool = True,
        classification: bool = False,
):
    # explain with TreeShapIQ observational --------------------------------------------------------
    print("\nTreeShapIQ explanations (observational) ------------------")
    explanation_time, empty_prediction = 0, 0.
    sii_values_dict: dict[int, np.ndarray] = {order: 0. for order in range(1, max_interaction_order + 1)}  # will be filled with the SII values for each interaction order

    if type(tree_model) != list:
        tree_model = [tree_model]

    for i, tree_model in tqdm(enumerate(tree_model), total=len(tree_model)):
        explainer = TreeShapIQ(
            tree_model=tree_model,
            max_interaction_order=max_interaction_order,
            n_features=n_features,
            observational=observational,
            background_dataset=background_dataset
        )
        start_time = time.time()
        explanation_scores: dict[int, np.ndarray] = explainer.explain(x=x_explain)
        explanation_time += time.time() - start_time
        sii_values_dict = {
            order: sii_values_dict[order] + explanation_scores[order]
            for order in range(1, max_interaction_order + 1)
        }
        empty_prediction += explainer.empty_prediction
    print("Time taken", explanation_time)
    explanation_sum = np.sum(list(sii_values_dict[1])) + empty_prediction
    if classification:  # we assume that every classifier uses logit as link function
        explanation_sum = expit(explanation_sum)
    print(sii_values_dict[1])
    print("Empty prediction", empty_prediction)
    print("Sum", explanation_sum)

    return sii_values_dict, empty_prediction


def run_main_experiment(
        model,
        x_explain: np.ndarray,
        y_true_label: float,
        explanation_id: int,
        max_interaction_order: int,
        n_features: int,
        feature_names: list[str],
        dataset_name: str,
        background_dataset: np.ndarray = None,
        observational: bool = True,
        save_figures: bool = False,
        classification: bool = False,
        show_plots: bool = True,
        force_limits: tuple[float, float] = None,
        sv_dim: int = None,
        output_type: str = "raw",
        model_flag: str = None,
        class_label: int = None,
) -> None:

    title: str = f"{dataset_name}: "
    save_name: str = f"plots/{dataset_name.lower().replace(' ', '_')}"
    if model_flag is not None:
        save_name += f"_{model_flag}"
        title = f"{dataset_name} {model_flag}: "
    save_name += f"_instance_{explanation_id}_order_{max_interaction_order}"

    # create feature names
    feature_names_abbrev = []
    for feature_name in feature_names:
        small_feature_name = feature_name[:5]
        # if the 5th character is a ' ' or '_' or '-' we want to extend the name by two characters
        if small_feature_name[-1] in [" ", "_", "-"]:
            small_feature_name += feature_name[5]
        feature_names_abbrev.append(small_feature_name + ".")

    #feature_names_abbrev = [feature[:5] + "." for feature in feature_names]
    feature_names_values = [feature + f"\n({round(x_explain[i], 2)})"
                            for i, feature in enumerate(feature_names_abbrev)]

    # get model output and true label --------------------------------------------------------------

    try:
        model_output = model.predict(x_explain.reshape(1, -1))[0]
        model_output_proba = model.predict_proba(x_explain.reshape(1, -1))[0]
        try:
            model_output_logit = model.predict_log_proba(x_explain.reshape(1, -1))[0]
        except AttributeError:
            model_output_logit = logit(model_output_proba)
        print("Model output proba:", model_output_proba,
              "logit:", model_output_logit, "True label:", y_true_label)
    except AttributeError:  # if model is a regressor
        model_output = round(model.predict(x_explain.reshape(1, -1))[0], 2)
        y_true_label = round(y_true_label, 2)
        print("Model output", model_output, "True label", y_true_label)

    # convert the tree -----------------------------------------------------------------------------
    if class_label is None:
        class_label = None if not classification else 1
    tree_model = convert_tree_estimator(model, class_label=class_label, output_type=output_type)

    # explain with TreeShapIQ observational --------------------------------------------------------

    sii_values_dict, empty_prediction = run_shap_iq(
        tree_model=tree_model,
        x_explain=x_explain,
        max_interaction_order=max_interaction_order,
        n_features=n_features,
        background_dataset=background_dataset,
        observational=observational,
        classification=classification
    )

    # generate n-SII values ------------------------------------------------------------------------

    n_sii_values, n_sii_values_one_dim = transform_interactions_in_n_shapley(
        interaction_values=sii_values_dict,
        n=max_interaction_order,
        n_features=n_features,
    )
    n_shapley_values_pos, n_shapley_values_neg = n_sii_values_one_dim

    sum_value: float = empty_prediction
    for order in range(1, max_interaction_order + 1):
        sum_value += sum(list(n_shapley_values_pos[order].values()))
        sum_value += sum(list(n_shapley_values_neg[order].values()))
    print("Sum of n-Shapley values", sum_value)

    # explain with TreeShap ------------------------------------------------------------------------

    print("\nTreeShap explanations (observational) ------------------")
    if observational:
        explainer_shap = TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        explainer_shap = TreeExplainer(
            model, feature_perturbation="interventional", data=background_dataset[:50])
    # reshape x_explain in 2 dim matrix
    x_explain_sv = copy(x_explain.reshape(1, -1))
    sv_shap = explainer_shap.shap_values(x_explain_sv).copy()
    if sv_dim is not None:
        try:
            sv_shap = sv_shap[sv_dim]
        except IndexError:
            sv_shap = sv_shap
    try:
        shap_empty_pred = explainer_shap.expected_value[0]
    except IndexError:
        shap_empty_pred = explainer_shap.expected_value
    if len(sv_shap) == 1:
        sv_shap = sv_shap[0]
    explanation_sum: float = np.sum(sv_shap) + empty_prediction
    if classification:  # we assume that every classifier uses logit as link function
        explanation_sum = expit(explanation_sum)
    print(sv_shap)
    print("Empty prediction", empty_prediction)
    print("Sum", explanation_sum)

    # plot the n-SII values ------------------------------------------------------------------------

    fig_obs, axis_obs = plot_n_sii(
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
        feature_names=feature_names_abbrev,
        n_sii_order=max_interaction_order
    )
    axis_obs.set_title(title + f"n-SII plot for instance {explanation_id}")
    if save_figures:
        fig_obs.savefig(save_name + "_n_sii.pdf", bbox_inches="tight")
    fig_obs.show() if show_plots else plt.close("all")

    # plot the network plot ------------------------------------------------------------------------

    if max_interaction_order == 2:
        # plot the n-Shapley values
        fig_network, axis_network = draw_interaction_network(
            first_order_values=n_sii_values[1],
            second_order_values=n_sii_values[2],
            feature_names=feature_names_values,
            n_features=n_features,
        )
        title_network: str = title + f"n-SII network plot for instance {explanation_id}\n" \
                                     f"Model output: {str(round(model_output, 2))}, " \
                                     f"True label: {y_true_label}"
        axis_network.set_title(title_network)
        if save_figures:
            fig_network.subplots_adjust(bottom=0.01, top=0.9, left=0.05, right=0.9)
            fig_network.savefig(save_name + "_network.pdf", bbox_inches=None)
        fig_network.show() if show_plots else plt.close("all")

    # plot the force plots -------------------------------------------------------------------------

    # SV force plot
    force(shap_empty_pred, sv_shap, feature_names=feature_names_values,
          matplotlib=True, show=False, figsize=(20, 3))
    if force_limits is not None:
        plt.xlim(force_limits)
    if save_figures:
        plt.savefig(save_name + "_force_SV.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")

    # plot the n-SII values as force plot
    n_sii_values_sv: np.ndarray = n_sii_values[1].copy()
    single_feature_names = copy(feature_names_abbrev)

    interaction_feature_names = []
    interaction_values = []
    interaction_feature_values = []
    interaction_feature_names_with_values = []
    for order in range(2, max_interaction_order + 1):
        for interaction in powerset(set(range(n_features)), min_size=order, max_size=order):
            comb_name: str = ""
            interaction_feature_value = ""
            for feature in interaction:
                if comb_name != "":
                    comb_name += " x "
                    interaction_feature_value += " x "
                comb_name += single_feature_names[feature]
                interaction_feature_value += f"{round(x_explain[feature], 2)}"
            interaction_values.append(n_sii_values[order][interaction])
            interaction_feature_names.append(comb_name)
            interaction_feature_values.append(interaction_feature_value)
            interaction_feature_names_with_values.append(comb_name + f"\n({interaction_feature_value})")
    interaction_values = np.asarray(interaction_values)
    interaction_feature_values = np.asarray(interaction_feature_values)

    # combine
    all_feature_names = single_feature_names + interaction_feature_names
    all_n_sii_values = np.concatenate([n_sii_values_sv, interaction_values])
    all_interaction_feature_values = np.concatenate([
        np.round(x_explain, 2), interaction_feature_values])
    all_feature_names_with_values = feature_names_values + interaction_feature_names_with_values

    # plot the TreeShap values

    force(shap_empty_pred, all_n_sii_values, feature_names=all_feature_names_with_values,
          matplotlib=True, show=False, figsize=(20, 3))
    if force_limits is not None:
        plt.xlim(force_limits)
    if save_figures:
        plt.savefig(save_name + "_force_n_SII.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")

    # plot the waterfall plot ----------------------------------------------------------------------

    # SV waterfall plot
    shap_explanation = Explanation(
        values=sv_shap,
        base_values=shap_empty_pred,
        data=x_explain,
        feature_names=feature_names_abbrev
    )
    waterfall(shap_explanation, show=False)
    if save_figures:
        plt.savefig(save_name + "_waterfall_SV.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")

    # plot the n-SII values as waterfall plot
    n_sii_explanation = Explanation(
        values=all_n_sii_values,
        base_values=shap_empty_pred,
        data=all_interaction_feature_values,
        feature_names=all_feature_names
    )
    waterfall(n_sii_explanation, show=False)
    if save_figures:
        plt.savefig(save_name + "_waterfall_n_sii.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")
