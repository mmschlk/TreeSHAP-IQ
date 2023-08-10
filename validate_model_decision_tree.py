import time

from scipy.special import binom

from tree_shap_iq.conversion import convert_tree_estimator
from tree_shap_iq.base import TreeShapIQ
from tree_shap_iq.old import TreeShapIQ as TreeSHAPIQ_gt

if __name__ == "__main__":
    DO_TREE_SHAP = True
    DO_PLOTTING = False
    DO_OBSERVATIONAL = True
    DO_GROUND_TRUTH = True

    INTERACTION_ORDER = 2
    INTERACTION_TYPE = "SII"

    if DO_TREE_SHAP:
        try:
            from shap import TreeExplainer
        except ImportError:
            print("TreeSHAP not available. Please install shap package.")
            DO_TREE_SHAPE = False

    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    import matplotlib.pyplot as plt
    import numpy as np
    from copy import deepcopy

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    X, y = make_regression(1000, n_features=12)
    clf = DecisionTreeRegressor(max_depth=10, random_state=random_seed).fit(X, y)

    x_input = X[2:3]
    print("Output f(x):", clf.predict(x_input)[0])

    if DO_PLOTTING:
        plt.figure(dpi=150)
        plot_tree(clf, node_ids=True, proportion=True)
        plt.savefig("tree.pdf")

    # TreeSHAP -------------------------------------------------------------------------------------

    my_thresholds = clf.tree_.threshold.copy()

    my_features = clf.tree_.feature.copy()

    tree_model = convert_tree_estimator(clf)

    # LinearTreeSHAP -------------------------------------------------------------------------------
    # explain the tree with LinearTreeSHAP

    explainer = TreeShapIQ(
        tree_model=deepcopy(tree_model), n_features=x_input.shape[1], observational=True,
        max_interaction_order=INTERACTION_ORDER, interaction_type=INTERACTION_TYPE

    )
    start_time = time.time()
    sv_linear_tree_shap = explainer.explain(x_input[0], INTERACTION_ORDER)
    time_elapsed = time.time() - start_time

    print("Linear")
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - SVs (obs.)      ", sv_linear_tree_shap)
    print("Linear - sum SVs (obs.)  ", sv_linear_tree_shap[2].sum() + explainer.empty_prediction)
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - empty pred      ", explainer.empty_prediction)
    print()

    if not DO_OBSERVATIONAL:
        start_time = time.time()
        explainer = TreeShapIQ(
            tree_model=deepcopy(tree_model),
            n_features=x_input.shape[1],
            observational=False,
            background_dataset=X,
            max_interaction_order=INTERACTION_ORDER,
            interaction_type=INTERACTION_TYPE
        )
        sv_linear_tree_shap = explainer.explain(x_input[0], INTERACTION_ORDER)
        time_elapsed = time.time() - start_time

        print("Linear")
        print("Linear - time elapsed    ", time_elapsed)
        print("Linear - SVs (int.)      ", sv_linear_tree_shap)
        print("Linear - sum SVs (int.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
        print("Linear - time elapsed    ", time_elapsed)
        print("Linear - empty pred      ", explainer.empty_prediction)
        print()

    # Ground Truth Brute Force ---------------------------------------------------------------------
    # compute the ground truth interactions with brute force
    if DO_GROUND_TRUTH:
        explainer_gt = TreeSHAPIQ_gt(
        tree_model=deepcopy(tree_model), n_features=x_input.shape[1], observational=True,
        max_interaction_order=INTERACTION_ORDER, interaction_type=INTERACTION_TYPE
        )
        start_time = time.time()
        gt_results = explainer_gt.explain_brute_force(x_input[0], INTERACTION_ORDER)
        ground_truth_shap_int, ground_truth_shap_int_pos = gt_results
        time_elapsed = time.time() - start_time

        print("Ground Truth")
        print("GT - time elapsed        ", time_elapsed)
        print()
        for ORDER in range(1, INTERACTION_ORDER + 1):
            print(f"GT - order {ORDER} SIs         ", ground_truth_shap_int[ORDER])
            print(f"GT - order {ORDER} sum SIs     ", ground_truth_shap_int[ORDER].sum())
            print()

        # debug order =2
        if len(sv_linear_tree_shap) == binom(x_input.shape[1], 2):
            mismatch = np.where((ground_truth_shap_int[2] - sv_linear_tree_shap) ** 2 > 0.001)
            for key in mismatch[0]:
                print(ground_truth_shap_int_pos[2][key])

    if DO_TREE_SHAP:
        # explain the tree with observational TreeSHAP
        start_time = time.time()
        if DO_OBSERVATIONAL:
            explainer_shap = TreeExplainer(deepcopy(clf),
                                           feature_perturbation="tree_path_dependent")
        else:
            explainer_shap = TreeExplainer(deepcopy(clf), feature_perturbation="interventional",
                                           data=X[:1000])

        sv_tree_shap = explainer_shap.shap_values(x_input)
        time_elapsed = time.time() - start_time

        print("TreeSHAP")
        print("TreeSHAP - time elapsed  ", time_elapsed)
        print("TreeSHAP - SVs           ", sv_tree_shap)
        print("TreeSHAP - sum SVs       ", sv_tree_shap.sum() + explainer_shap.expected_value)
        print("TreeSHAP - empty pred    ", explainer_shap.expected_value)
        print()
