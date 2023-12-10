"""This module is for plotting the run-time experiment results."""
import copy
import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if __name__ == "__main__":
    # Path to the CSV file
    file_name: str = "run_time.csv"
    path = os.path.join(["..", file_name])

    plot_name: str = "run_time_plot"
    col_names = ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
                 "n_decision_nodes", "n_leaves", "leaves_times_depth", "elapsed_time"]

    run_time_df = pd.read_csv(path)

    # get mean and std for each max_interaction_order of elapsed time
    run_time_df = run_time_df.groupby(
        ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
         "n_decision_nodes", "n_leaves", "leaves_times_depth"]).agg(
        {"elapsed_time": ["mean", "std"]}
    ).reset_index()

    # rename the grouped columns with "_" in between elasped_time and mean/std
    new_col_names = col_names[:-1] + ["elapsed_time_mean", "elapsed_time_std"]
    run_time_df.columns = copy.deepcopy(new_col_names)

    #Plots for changing tree complexity
    interaction_order = 2
    plot_df = run_time_df[run_time_df["interaction_order"] == interaction_order]

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(plot_df["leaves_times_depth"], plot_df["elapsed_time_mean"])

    x_axis_feature = "leaves_times_depth"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_id in plot_df["model_id"].unique():
        model_df = plot_df[plot_df["model_id"] == model_id]
        ax.errorbar(
            model_df[x_axis_feature], model_df["elapsed_time_mean"],
            yerr=model_df["elapsed_time_std"],
            label=model_id, marker="o"
        )
        # add std as confidence band
        ax.fill_between(
            model_df[x_axis_feature],
            model_df["elapsed_time_mean"] - model_df["elapsed_time_std"],
            model_df["elapsed_time_mean"] + model_df["elapsed_time_std"],
            alpha=0.2
        )
    ax.set_xlabel("#Leaves x Depth")
    ax.set_ylabel("Elapsed Time (s)")
    ax.set_title("Computation time for DT with increasing #Leaves x Depth")
    ax.legend()
    # Adding a textbox with additional information
    info_text = (
        f"Pearson Correlation: "+str(round(correlation_coefficient,4))+"\n"
        f"p-value: "+str(round(p_value,4))
    )
    #ax.text(0.05, 0.75, info_text, transform=ax.transAxes, fontsize=10,
    #        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    plt.show()
    fig.savefig(plot_name+"_tree_"+str(interaction_order)+"_leaves_x_depth", dpi=300, bbox_inches='tight')



    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(plot_df["n_nodes"], plot_df["elapsed_time_mean"])

    x_axis_feature = "n_nodes"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_id in plot_df["model_id"].unique():
        model_df = plot_df[plot_df["model_id"] == model_id]
        ax.errorbar(
            model_df[x_axis_feature], model_df["elapsed_time_mean"],
            yerr=model_df["elapsed_time_std"],
            label=model_id, marker="o"
        )
        # add std as confidence band
        ax.fill_between(
            model_df[x_axis_feature],
            model_df["elapsed_time_mean"] - model_df["elapsed_time_std"],
            model_df["elapsed_time_mean"] + model_df["elapsed_time_std"],
            alpha=0.2
        )
    ax.set_xlabel("#Nodes")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    # Adding a textbox with additional information
    info_text = (
        f"Pearson Correlation: "+str(round(correlation_coefficient,4))+"\n"
        f"p-value: "+str(round(p_value,4))
    )
    ax.text(0.05, 0.75, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

    plt.show()
    fig.savefig(plot_name+"_tree_"+str(interaction_order)+"_nodes", dpi=300, bbox_inches='tight')




    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(plot_df["depth"], plot_df["elapsed_time_mean"])

    x_axis_feature = "depth"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_id in plot_df["model_id"].unique():
        model_df = plot_df[plot_df["model_id"] == model_id]
        ax.errorbar(
            model_df[x_axis_feature], model_df["elapsed_time_mean"],
            yerr=model_df["elapsed_time_std"],
            label=model_id, marker="o"
        )
        # add std as confidence band
        ax.fill_between(
            model_df[x_axis_feature],
            model_df["elapsed_time_mean"] - model_df["elapsed_time_std"],
            model_df["elapsed_time_mean"] + model_df["elapsed_time_std"],
            alpha=0.2
        )
    ax.set_title("Computation time for DT with increasing Depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    # Adding a textbox with additional information
    info_text = (
        f"Pearson Correlation: "+str(round(correlation_coefficient,4))+"\n"
        f"p-value: "+str(round(p_value,4))
    )
    #ax.text(0.05, 0.75, info_text, transform=ax.transAxes, fontsize=10,
    #        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

    plt.show()
    fig.savefig(plot_name+"_tree_"+str(interaction_order)+"_depth", dpi=300, bbox_inches='tight')

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(plot_df["n_leaves"], plot_df["elapsed_time_mean"])


    x_axis_feature = "n_leaves"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_id in plot_df["model_id"].unique():
        model_df = plot_df[plot_df["model_id"] == model_id]
        ax.errorbar(
            model_df[x_axis_feature], model_df["elapsed_time_mean"],
            yerr=model_df["elapsed_time_std"],
            label=model_id, marker="o"
        )
        # add std as confidence band
        ax.fill_between(
            model_df[x_axis_feature],
            model_df["elapsed_time_mean"] - model_df["elapsed_time_std"],
            model_df["elapsed_time_mean"] + model_df["elapsed_time_std"],
            alpha=0.2
        )
    ax.set_title("Computation time for DT with increasing #Leaves")
    ax.set_xlabel("#Leaves")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    # Adding a textbox with additional information
    info_text = (
        f"Pearson Correlation: "+str(round(correlation_coefficient,4))+"\n"
        f"p-value: "+str(round(p_value,4))
    )
    ax.text(0.05, 0.75, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    plt.show()
    fig.savefig(plot_name+"_tree_"+str(interaction_order)+"_leaves", dpi=300, bbox_inches='tight')



    plot_df = run_time_df[run_time_df["depth"] == 14]

    """
    x_axis_feature = "interaction_order"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_id in plot_df["model_id"].unique():
        model_df = plot_df[plot_df["model_id"] == model_id]
        ax.errorbar(
            model_df[x_axis_feature], model_df["elapsed_time_mean"],
            yerr=model_df["elapsed_time_std"],
            label=model_id, marker="o"
        )
        # add std as confidence band
        ax.fill_between(
            model_df[x_axis_feature],
            model_df["elapsed_time_mean"] - model_df["elapsed_time_std"],
            model_df["elapsed_time_mean"] + model_df["elapsed_time_std"],
            alpha=0.2
        )
    ax.set_xlabel("interaction order")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    plt.show()
    """


    #Plot by interaction order
    tree_depth: str = 20
    file_name: str = "run_time_interaction_20.csv"

    col_names = ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
                 "n_decision_nodes", "n_leaves", "leaves_times_depth", "elapsed_time"]

    run_time_df = pd.read_csv(path+file_name)

    # get mean and std for each max_interaction_order of elapsed time
    run_time_df = run_time_df.groupby(
        ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
         "n_decision_nodes", "n_leaves", "leaves_times_depth"]).agg(
        {"elapsed_time": ["mean", "std"]}
    ).reset_index()

    # rename the grouped columns with "_" in between elasped_time and mean/std
    new_col_names = col_names[:-1] + ["elapsed_time_mean", "elapsed_time_std"]
    run_time_df.columns = copy.deepcopy(new_col_names)

    plot_df = run_time_df[run_time_df["depth"] == tree_depth]

    x_axis_feature = "interaction_order"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_id in plot_df["model_id"].unique():
        model_df = plot_df[plot_df["model_id"] == model_id]
        ax.errorbar(
            model_df[x_axis_feature], model_df["elapsed_time_mean"],
            yerr=model_df["elapsed_time_std"],
            label=model_id, marker="o"
        )
        # add std as confidence band
        ax.fill_between(
            model_df[x_axis_feature],
            model_df["elapsed_time_mean"] - model_df["elapsed_time_std"],
            model_df["elapsed_time_mean"] + model_df["elapsed_time_std"],
            alpha=0.2
        )

    ax.set_title("Computation time for DT with increasing Interaction Order")
    ax.set_xlabel("Interaction Order")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    plt.show()
    fig.savefig(plot_name+"_int_"+str(tree_depth)+"_order", dpi=300, bbox_inches='tight')

