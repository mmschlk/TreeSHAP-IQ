"""This module is for plotting the run-time experiment results."""
import copy

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file_name: str = "run_time/run_time.csv"
    depth = 25
    file_name_interaction: str = f"run_time_interaction_{depth}.csv"


    col_names = ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
                 "n_decision_nodes", "n_leaves", "leaves_times_depth", "elapsed_time"]

    run_time_df = pd.read_csv(file_name)

    # get mean and std for each max_interaction_order of elapsed time
    run_time_df = run_time_df.groupby(
        ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
                 "n_decision_nodes", "n_leaves", "leaves_times_depth"]).agg(
        {"elapsed_time": ["mean", "std"]}
    ).reset_index()

    # rename the grouped columns with "_" in between elasped_time and mean/std
    new_col_names = col_names[:-1] + ["elapsed_time_mean", "elapsed_time_std"]
    run_time_df.columns = copy.deepcopy(new_col_names)

    plot_df = run_time_df[run_time_df["interaction_order"] == 2]

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
    ax.set_xlabel("#leaves * depth")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    plt.show()

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
    ax.set_xlabel("#nodes")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    plt.show()

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
    ax.set_xlabel("depth")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    plt.show()

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
    ax.set_xlabel("# leaves")
    ax.set_ylabel("Elapsed Time (s)")
    ax.legend()
    plt.show()

    run_time_df = pd.read_csv(file_name_interaction)

    # get mean and std for each max_interaction_order of elapsed time
    run_time_df = run_time_df.groupby(
        ["model_id", "n_features", "interaction_order", "depth", "n_nodes",
         "n_decision_nodes", "n_leaves", "leaves_times_depth"]).agg(
        {"elapsed_time": ["mean", "std"]}
    ).reset_index()

    # rename the grouped columns with "_" in between elasped_time and mean/std
    new_col_names = col_names[:-1] + ["elapsed_time_mean", "elapsed_time_std"]
    run_time_df.columns = copy.deepcopy(new_col_names)

    plot_df = run_time_df[run_time_df["depth"] == depth]

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
