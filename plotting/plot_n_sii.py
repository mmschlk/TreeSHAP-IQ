"""This module contains functions to plot the n_sii values for a given instance."""
__all__ = ["plot_n_sii", "transform_interactions_in_n_shapley"]

from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.special import bernoulli

from tree_shap_iq.utils import powerset

COLORS_N_SII = ["#D81B60", "#FFB000", "#1E88E5", "#FE6100", "#7F975F", "#74ced2", "#708090", "#9966CC", "#CCCCCC", "#800080"]
COLORS_N_SII = COLORS_N_SII * (5 + (len(COLORS_N_SII)))  # repeat colors


def _generate_interactions_lookup(
        n_features: int,
        max_order: int
):
    """Generates a lookup table for the interaction scores from subset to index."""
    counter_interaction = 0  # stores position of interactions
    shapley_interactions_lookup: dict = {}
    for S in powerset(range(n_features), max_order):
        shapley_interactions_lookup[S] = counter_interaction
        counter_interaction += 1
    return shapley_interactions_lookup


def _convert_n_shapley_values_to_one_dimension(
        n_shapley_values: dict[int, np.ndarray],
        n_features: int,
        n: int = None
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Converts the n-Shapley values to one dimension

    Args:
        n_shapley_values (dict[int, np.ndarray]): The n-Shapley values
        n_features (int): The number of features.
        n (int, optional): The order of the Shapley values. Defaults to None.

    Returns:
        tuple[dict[int, np.ndarray], dict[int, np.ndarray]]: A tuple containing the positive and
            negative parts of the n-Shapley values
    """
    if n is None:
        n = max(n_shapley_values.keys())
    N = set(range(n_features))
    result_pos = {order: {player: 0. for player in range(n_features)} for order in range(1, n + 1)}
    result_neg = {order: {player: 0. for player in range(n_features)} for order in range(1, n + 1)}
    result_values = {order: {player: [] for player in range(n_features)} for order in range(1, n + 1)}

    for S in powerset(N, min_size=1, max_size=n):
        n_shap_value = n_shapley_values[len(S)][tuple(S)]
        for player in S:
            result_values[len(S)][player].append(n_shap_value)

    for S in powerset(N, min_size=1, max_size=n):
        n_shap_value = n_shapley_values[len(S)][tuple(S)]
        for player in S:
            if n_shap_value > 0:
                result_pos[len(S)][player] += n_shap_value / len(S)
            if n_shap_value < 0:
                result_neg[len(S)][player] += n_shap_value / len(S)
    return result_pos, result_neg


def _transform_shape_of_interaction_values(
        interaction_values: dict[int, np.ndarray],
        n_features: int,
) -> dict[int, np.ndarray]:
    """Transforms the shape of the interaction values from a vector of length
        binom(n_features, order) to a tensor of shape (n_features, ..., n_features).
    """
    result = {}
    for order, values in interaction_values.items():
        result[order] = np.zeros(np.repeat(n_features, order))
        for idx, value in enumerate(values):
            result[order][tuple(np.unravel_index(idx, result[order].shape))] = value
    return result


def transform_interactions_in_n_shapley(
        n_features: int,
        n: int,
        interaction_values: dict[int, np.ndarray],
        reduce_one_dimension: bool = False
) -> Union[dict[int, np.ndarray], tuple[dict[int, np.ndarray], dict[int, np.ndarray]]]:
    """Computes the n-Shapley values from the interaction values

    Args:
        n_features (int): The number of features.
        interaction_values (Dict[int, np.ndarray], optional): The interaction values.
            Defaults to None.
        n int: The order of the interaction values.
        reduce_one_dimension (bool, optional): If True, the n-Shapley values are reduced to one
            dimension. Defaults to False.

    Returns:
        dict: A dictionary containing the n-Shapley values
    """

    def init_results():
        """Initialize the results dictionary with zero arrays.

        Returns:
            dict[int, np.ndarray]: Dictionary with zero arrays for each interaction order.
            dict[tuple, int]: Dictionary with the index of each interaction.
        """
        result_dict, lookup_dict = {}, {}
        for k in range(1, n + 1):
            result_dict[k] = np.zeros(np.repeat(n_features, k))
            lookup_dict[k] = _generate_interactions_lookup(n_features, k)
        return result_dict, lookup_dict

    N = set(range(n_features))
    bernoulli_numbers = bernoulli(n)
    result, lookup = init_results()
    # all subsets S with 1 <= |S| <= n
    for S in powerset(N, min_size=1, max_size=n):
        # get un-normalized interaction value (delta_S(x))
        S_index = lookup[len(S)][tuple(S)]
        S_effect = interaction_values[len(S)][S_index]
        subset_size = len(S)
        # go over all subsets T of length |S| + 1, ..., n that contain S
        for T in powerset(N, min_size=subset_size + 1, max_size=n):
            if not set(S).issubset(T):
                continue
            # get the effect of T
            T_index = lookup[len(T)][tuple(T)]
            T_effect = interaction_values[len(T)][T_index]
            # normalization with bernoulli numbers
            S_effect += bernoulli_numbers[len(T) - subset_size] * T_effect
        result[len(S)][tuple(S)] = S_effect
    if not reduce_one_dimension:
        return result
    return _convert_n_shapley_values_to_one_dimension(result, n=n, n_features=n_features)


def plot_n_sii(
        feature_names: np.ndarray,
        n_shapley_values_pos: dict,
        n_shapley_values_neg: dict,
        n_sii_order: int
):
    """Plot the n-SII values for a given instance.

    Args:
        feature_names (list): The names of the features.
        n_shapley_values_pos (dict): The positive n-SII values.
        n_shapley_values_neg (dict): The negative n-SII values.
        n_sii_order (int): The order of the n-SII values.
    """

    params = {
        'legend.fontsize': 'x-large', 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'
    }
    fig, axis = plt.subplots(figsize=(6, 4.15))

    # transform data to make plotting easier
    n_features = len(feature_names)
    x = np.arange(n_features)
    values_pos = []
    for order, values in n_shapley_values_pos.items():
        values_pos.append(values)
    values_pos = pd.DataFrame(values_pos)
    values_neg = []
    for order, values in n_shapley_values_neg.items():
        values_neg.append(values)
    values_neg = pd.DataFrame(values_neg)

    # get helper variables for plotting the bars
    min_max_values = [0, 0]  # to set the y-axis limits after all bars are plotted
    reference_pos = np.zeros(n_features)  # to plot the bars on top of each other
    reference_neg = deepcopy(np.asarray(values_neg.loc[0]))  # to plot the bars below of each other

    # plot the bar segments
    for order in range(len(values_pos)):
        axis.bar(
            x,
            height=values_pos.loc[order],
            bottom=reference_pos,
            color=COLORS_N_SII[order]
        )
        axis.bar(
            x,
            height=abs(values_neg.loc[order]),
            bottom=reference_neg,
            color=COLORS_N_SII[order]
        )
        axis.axhline(y=0, color="black", linestyle="solid")
        reference_pos += values_pos.loc[order]
        try:
            reference_neg += values_neg.loc[order + 1]
        except KeyError:
            pass
        min_max_values[0] = min(min_max_values[0], min(reference_neg))
        min_max_values[1] = max(min_max_values[1], max(reference_pos))

    # add a legend to the plots
    legend_elements = []
    for order in range(n_sii_order):
        legend_elements.append(
            Patch(facecolor=COLORS_N_SII[order], edgecolor='black', label=f"Order {order + 1}"))
    axis.legend(handles=legend_elements, loc='upper center', ncol=min(n_sii_order, 4))

    x_ticks_labels = [feature for feature in feature_names]  # might be unnecessary
    axis.set_xticks(x)
    axis.set_xticklabels(x_ticks_labels, rotation=45, ha='right')

    axis.set_xlim(-0.5, n_features - 0.5)
    axis.set_ylim(min_max_values[0] * 1.05, min_max_values[1] * 1.3)

    axis.set_ylabel("n-SII values")
    axis.set_xlabel("features")
    axis.set_title(f"n-SII values up to order ${n_sii_order}$")

    plt.tight_layout()

    return fig, axis
