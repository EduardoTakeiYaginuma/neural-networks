# Activity 3 — Full source code

The three files below reproduce the complete activity. The MLP is implemented
with NumPy only; scikit-learn is used solely for the permitted data generation,
splits, baselines and confusion matrix.

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python docs/exercises/mlp/code/run_report.py
```

## `code/mlp.py` — hand-written, reusable MLP

```python
--8<-- "docs/exercises/mlp/code/mlp.py"
```

## `code/run_report.py` — experiments and all six figures

```python
--8<-- "docs/exercises/mlp/code/run_report.py"
```

## `code/style.py` — plotting style

```python
--8<-- "docs/exercises/mlp/code/style.py"
```

## Machine-readable results

All reported values are preserved in
[`results/mlp/results.json`](https://github.com/EduardoTakeiYaginuma/neural-networks/blob/main/results/mlp/results.json).
