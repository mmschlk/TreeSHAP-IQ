import numba
import time

import numpy as np
from collections import namedtuple
import scipy.special as sp

Tree = namedtuple('Tree', 'weights,leaf_predictions,parents,edge_heights,features,children_left,children_right,thresholds,max_depth,num_nodes')


def copy_tree(tree):
    weights = np.ones_like(tree.threshold)
    leaf_predictions = np.zeros_like(tree.threshold)
    parents = np.full_like(tree.children_left, -1)
    edge_heights = np.zeros_like(tree.children_left)

    def _recursive_copy(node=0, feature=None,
                        parent_samples=None, prod_weight=1.0,
                        seen_features=dict()):
        n_sample, child_left, child_right = (tree.n_node_samples[node],
                                             tree.children_left[node], tree.children_right[node])
        if feature is not None:
            weight = n_sample / parent_samples
            prod_weight *= weight
            if feature in seen_features:
                parents[node] = seen_features[feature]
                weight *= weights[seen_features[feature]]
            weights[node] = weight
            seen_features[feature] = node
        if child_left >= 0:  # not leaf
            left_max_features = _recursive_copy(child_left, tree.feature[node], n_sample,
                                                prod_weight, seen_features.copy())
            right_max_features = _recursive_copy(child_right, tree.feature[node], n_sample,
                                                 prod_weight, seen_features.copy())
            edge_heights[node] = max(left_max_features, right_max_features)
            return edge_heights[node]
        else:  # is leaf
            edge_heights[node] = len(seen_features)
            return edge_heights[node]

    _recursive_copy()
    return Tree(weights, tree.n_node_samples / tree.n_node_samples[0] * tree.value.ravel(), parents,
                edge_heights, tree.feature, tree.children_left, tree.children_right, tree.threshold,
                tree.max_depth, tree.children_left.shape[0])


def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])


def get_N(max_size=10):
    N = np.zeros((max_size, max_size))
    for i in range(max_size):
        N[i, :i + 1] = get_norm_weight(i)
    return N


def get_N_prime(max_size=10):
    N = np.zeros((max_size + 2, max_size + 2))
    for i in range(max_size + 2):
        N[i, :i + 1] = get_norm_weight(i)
    N_prime = np.zeros((max_size + 2, max_size + 2))
    for i in range(max_size + 2):
        N_prime[i, :i + 1] = N[:i + 1, :i + 1].dot(1 / N[i, :i + 1])
    return N_prime


def get_N_v2(D):
    depth = D.shape[0]
    Ns = np.zeros((depth + 1, depth))
    for i in range(1, depth + 1):
        Ns[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(1. / get_norm_weight(i - 1))
    return Ns


np.seterr(divide='raise')

def psi(E, D_power, D, q, Ns, d):
    n = Ns[d, :d]
    return ((E*D_power/(D+q))[:d]).dot(n)/d

def _inference(weights,
              leaf_predictions,
              parents,
              edge_heights,
              features,
              children_left,
              children_right,
              thresholds,
              max_depth,
              x,
              activation,
              result, D_powers,
              D, Ns, C, E, node=0,
              edge_feature=-1, depth=0):

    left, right, parent, child_edge_feature = (
                            children_left[node],
                            children_right[node],
                            parents[node],
                            features[node]
                            )
    left_height, right_height, parent_height, current_height = (
                            edge_heights[left],
                            edge_heights[right],
                            edge_heights[parent],
                            edge_heights[node]
                            )
    if left >= 0:
        if x[child_edge_feature] <= thresholds[node]:
            activation[left], activation[right] = True, False
        else:
            activation[left], activation[right] = False, True

    if edge_feature >= 0:
        if parent >= 0:
            activation[node] &= activation[parent]

        if activation[node]:
            q_eff = 1./weights[node]
        else:
            q_eff = 0.
        C[depth] = C[depth-1]*(D+q_eff)

        if parent >= 0:
            if activation[parent]:
                s_eff = 1./weights[parent]
            else:
                s_eff = 0.
            C[depth] = C[depth]/(D+s_eff)
    if left < 0:
        E[depth] = C[depth]*leaf_predictions[node]
    else:
        _inference(weights,
                  leaf_predictions,
                  parents,
                  edge_heights,
                  features,
                  children_left,
                  children_right,
                  thresholds,
                  max_depth,
                  x,
                  activation,
                  result, D_powers,
                  D, Ns, C, E, left,
                  child_edge_feature,
                  depth+1
                  )
        E[depth] = E[depth+1]*D_powers[current_height-left_height]
        _inference(weights,
                  leaf_predictions,
                  parents,
                  edge_heights,
                  features,
                  children_left,
                  children_right,
                  thresholds,
                  max_depth,
                  x,
                  activation,
                  result, D_powers,
                  D, Ns, C, E, right,
                  child_edge_feature,
                  depth+1
                  )
        E[depth] += E[depth+1]*D_powers[current_height-right_height]


    if edge_feature >= 0:
        value = (q_eff-1)*psi(E[depth], D_powers[0], D, q_eff, Ns, current_height)
        result[edge_feature] += value
        if parent >= 0:
            value = (s_eff-1)*psi(E[depth], D_powers[parent_height-current_height], D, s_eff, Ns, parent_height)
            result[edge_feature] -= value

def fast_inference(tree, D, D_powers, Ns, result, activation, x, max_depth, C, E):
    for i in range(x.shape[0]):
        _inference(tree.weights,
                   tree.leaf_predictions,
                   tree.parents,
                   tree.edge_heights,
                   tree.features,
                   tree.children_left,
                   tree.children_right,
                   tree.thresholds,
                   max_depth,
                   x[i],
                   activation,
                   result[i], D_powers,
                   D, Ns, C, E)

def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])

def get_N(D):
    depth = D.shape[0]
    Ns = np.zeros((depth+1, depth))
    for i in range(1, depth+1):
        Ns[i,:i] = np.linalg.inv(np.vander(D[:i]).T).dot(1./get_norm_weight(i-1))
    return Ns

def cache(D):
    return np.vander(D+1).T[::-1]

def inference(tree, x):
    D = np.polynomial.chebyshev.chebpts2(tree.max_depth)
    D_powers = cache(D)
    Ns = get_N(D)
    activation = np.zeros_like(tree.children_left, dtype=bool)
    C = np.zeros((tree.max_depth+1, tree.max_depth))
    E = np.zeros((tree.max_depth+1, tree.max_depth))
    C[0, :] = 1
    result = np.zeros_like(x)
    fast_inference(tree, D, D_powers, Ns, result, activation, x, tree.max_depth, C, E)
    result = np.zeros_like(x)
    start = time.time()
    fast_inference(tree, D, D_powers, Ns, result, activation, x, tree.max_depth, C, E)
    print('mine', time.time()-start)
    return result


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, export_text
    from shap import TreeExplainer as Truth
    import numpy as np
    np.random.seed(10)
    random_seed = 10
    x, y = make_regression(1000, n_features=15)
    clf = DecisionTreeRegressor(max_depth=50, random_state=random_seed).fit(x, y)
    sim = Truth(clf)
    mine_tree = copy_tree(clf.tree_)
    result = inference(mine_tree, x[:1])
    start = time.time()
    b = sim.shap_values(x[:1])
    print('b', time.time()-start)
    print('mine', result)
    print('b', b)
    np.testing.assert_array_almost_equal(result, b, 1)
