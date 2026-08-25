# Activity 1 — Full source code

Five files. `run_report.py` is the entry point: it creates the single
`np.random.default_rng(42)` used by Exercises 1 and 2, runs the three exercises in order,
writes the six figures to `docs/data/figures/` and every table and raw number to `results/`.

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python src/run_report.py
```

## `src/run_report.py` — entry point

```python
--8<-- "src/run_report.py"
```

## `src/ex1_point_clouds.py` — Exercise 1

```python
--8<-- "src/ex1_point_clouds.py"
```

## `src/ex2_nonlinearity.py` — Exercise 2

```python
--8<-- "src/ex2_nonlinearity.py"
```

## `src/ex3_realworld.py` — Exercise 3

```python
--8<-- "src/ex3_realworld.py"
```

## `src/style.py` — shared plotting style

```python
--8<-- "src/style.py"
```

## Raw results

Every number quoted in the report is also dumped to
[`results/results.json`](https://github.com/EduardoTakeiYaginuma/neural-networks/blob/main/results/results.json).
