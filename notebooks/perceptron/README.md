# Notebook — Activity 2 (`perceptron`)

The complete activity is presented in one self-contained notebook:

| Notebook | Contents |
|---|---|
| [`perceptron.ipynb`](perceptron.ipynb) | Hand-written perceptron, separable and overlapping datasets, pocket algorithm, Figures 1–6, analysis and results summary |

The notebook is saved with all outputs and reproduces the published
[Activity 2 report](../../docs/exercises/perceptron/index.md). It does not import
the report scripts: model, data generation, plotting and analysis are included
directly so it can be read and executed from beginning to end.

## Running it

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
pip install jupyterlab              # only needed to run the notebook
jupyter lab notebooks/perceptron/
```

The notebook creates exactly one `np.random.default_rng(42)` and runs Exercise 1
before Exercise 2. Keeping that order is necessary to reproduce every number in
the report.
