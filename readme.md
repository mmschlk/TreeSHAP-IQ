# Supplement Implementation for Linear Tree Shapley Interaction

This repository contains supplement implementation for the _Linear Tree Shapley Interaction_ project.
The goal is to apply the same notion of Linear TreeSHAP ([paper](https://papers.nips.cc/paper_files/paper/2022/hash/a5a3b1ef79520b7cd122d888673a3ebc-Abstract-Conference.html)) to interaction effects.

## Installation
The implementation was written and tested with python version 3.9.7.
Install the dependencies via pip:
```bash
pip install -r requirements.txt
```

Warning: If you want to compare the results with the original TreeSHAP implementation, you need to 
install the original TreeSHAP implementation from `pip` and change three lines of code in there.

Change line 250 in _tree.py from
```python
X_missing = np.isnan(X, dtype=np.bool)
```
to
```python
X_missing = np.isnan(X, dtype=bool)
``` 
and change line 1102 in _tree.py from
```python
X_missing = np.isnan(X, dtype=np.bool)
```
to
```python
X_missing = np.isnan(X, dtype=bool)
```
and change line 82 in _tabular.py from
```python
self._last_mask = np.zeros(data.shape[1], dtype=np.bool)
```
to
```python
self._last_mask = np.zeros(data.shape[1], dtype=bool)
```

## Experiemnts
Currently only initial experiments are present, which are described in the following list:
- _GradientBoostedTree_: An experiment on runnning a GradientBoostedTreeRegressor from sklearn is implemented in `boosted_tree_sklearn.py`.
