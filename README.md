# Neural Networks — Insper

Reports for the *Artificial Neural Networks and Deep Learning* course (Insper).

**Published site:** <https://eduardotakeiyaginuma.github.io/neural-networks/>

| Activity | Report |
|---|---|
| 1 — Data preparation and analysis | `docs/data/main.md` |

## Layout

```
src/run_report.py         entry point: one np.random.default_rng(42) for the whole report
src/ex1_point_clouds.py   Exercise 1 - 2D Gaussian clouds, separation ratio, mixing rate
src/ex2_nonlinearity.py   Exercise 2 - 5D shifted Gaussians vs concentric shells, PCA
src/ex3_realworld.py      Exercise 3 - Spaceship Titanic preprocessing for a tanh network
src/style.py              shared matplotlib style and the CVD-checked class palette
docs/                     MkDocs site (report pages + generated figures)
results/                  generated tables and results.json quoted by the report
```

## Reproducing

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

Exercise 3 uses the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
`train.csv`, which is **not** redistributed here. Download it and place it at `data/train.csv`
(8693 rows × 14 columns), then:

```bash
python src/run_report.py   # regenerates every figure in docs/data/figures and every table in results/
mkdocs serve               # preview at http://127.0.0.1:8000
```

Pushing to `main` builds and publishes the site through GitHub Actions
(`.github/workflows/deploy.yml`).
