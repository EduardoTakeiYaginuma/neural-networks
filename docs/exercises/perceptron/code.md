# Activity 2 — Full source code

Five files implement the complete experiment. `run_report.py` is the entry
point: it creates the single `np.random.default_rng(42)`, runs both exercises
in order, and regenerates all six figures plus the summary and raw results.

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python docs/exercises/perceptron/code/run_report.py
```

## `code/run_report.py` — entry point

```python
--8<-- "docs/exercises/perceptron/code/run_report.py"
```

## `code/perceptron.py` — hand-written model

```python
--8<-- "docs/exercises/perceptron/code/perceptron.py"
```

## `code/ex1_separable.py` — Exercise 1

```python
--8<-- "docs/exercises/perceptron/code/ex1_separable.py"
```

## `code/ex2_overlapping.py` — Exercise 2

```python
--8<-- "docs/exercises/perceptron/code/ex2_overlapping.py"
```

## `code/style.py` — plotting style

```python
--8<-- "docs/exercises/perceptron/code/style.py"
```

## Raw results

Every value quoted in the report is also stored in
[`results/perceptron/results.json`](https://github.com/EduardoTakeiYaginuma/neural-networks/blob/main/results/perceptron/results.json).
