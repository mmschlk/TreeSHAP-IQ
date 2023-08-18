# ⭐Supplement Implementation
**Paper: Beyond TreeSHAP: Efficient Computation of any-order Shapley Interactions for Tree Ensembles**

---

## 🔧Installation
The implementation was written and tested with python version 3.9.7.
Install the dependencies via pip:
```bash
pip install -r requirements.txt
```

## 📊Running the experiments

The main experiment functionality is implemented in `experiments_main.py` including all plots.
The scripts for running the experiments on the individual datasets are named `exp_<dataset>.py`.
Therein, you find the code for pre-processing, training, and explaining.


## 🔨Errors in outdated SHAP
Warning: If you want to compare the results with the original TreeSHAP implementation, you need to 
install the original TreeSHAP implementation from `pip` and change a couple of lines of code in there.

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
