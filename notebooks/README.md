# Notebooks — Activity 1

One notebook per exercise, each self-contained (no import from the report's `code/`) and saved with its
outputs, so it reads end to end without being run:

| Notebook | Exercise |
|---|---|
| [`ex1_point_clouds.ipynb`](ex1_point_clouds.ipynb) | 1 — Point clouds: geometry and spread in 2D |
| [`ex2_nonlinearity.ipynb`](ex2_nonlinearity.ipynb) | 2 — Non-linearity in higher dimensions |
| [`ex3_realworld.ipynb`](ex3_realworld.ipynb) | 3 — Preparing real-world data for a `tanh` network |

They are the same code and the same analysis as the published report
(<https://eduardotakeiyaginuma.github.io/neural-networks/exercises/data/>), and every number they
print matches it exactly. The report's figures are still produced by `python docs/exercises/data/code/run_report.py`;
the notebooks render theirs inline.

## Running them

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
pip install jupyterlab              # only needed to run the notebooks
jupyter lab notebooks/
```

Exercise 3 needs `data/train.csv` from the Kaggle
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition — the
file is not committed to the repository.

## One seed, two exercises

`docs/exercises/data/code/run_report.py` creates a single `np.random.default_rng(42)` and threads it through
Exercise 1 and then Exercise 2, so Exercise 2's draws depend on Exercise 1 having run first.
To reproduce the published numbers, `ex2_nonlinearity.ipynb` replays Exercise 1's 16 draws
before generating its own data (`REPLAY_EX1 = True` in the third cell). Set it to `False` for a
standalone run: the conclusions are identical, but the sampling-noise figures — notably the
Dataset II centre distance, which is pure noise around 0 — land on different values.
