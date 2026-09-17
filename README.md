# Neural Networks — Insper

Reports for the *Artificial Neural Networks and Deep Learning* course (Insper).
One repository for every activity of the course; the MkDocs site is the shared
part, everything else is namespaced by the activity slug.

**Published site:** <https://eduardotakeiyaginuma.github.io/neural-networks/>

| # | Activity | Slug | Report |
|---|---|---|---|
| 1 | Data preparation and analysis | `data` | [`docs/exercises/data/index.md`](docs/exercises/data/index.md) — published at [/exercises/data/](https://eduardotakeiyaginuma.github.io/neural-networks/exercises/data/) |
| 2 | Perceptron | `perceptron` | [`docs/exercises/perceptron/index.md`](docs/exercises/perceptron/index.md) — published at [/exercises/perceptron/](https://eduardotakeiyaginuma.github.io/neural-networks/exercises/perceptron/) |

## Layout

The slug is the contract. The course fixes four of them — `data`, `perceptron`,
`mlp`, `vae` — and each one appears as a folder in the same four places:

```
mkdocs.yml                              site config; nav has one section per activity
requirements.txt                        pipeline + site dependencies (shared)
docs/
  index.md                              landing page: the table of activities
  stylesheets/extra.css                 shared theme
  javascripts/mathjax.js                shared MathJax config
  exercises/
    <slug>/
      index.md                          the report (front matter: exercise, ai_use)
      code.md                           full source listing, pulled in by snippet
      code/                             the runnable scripts, one per exercise + an entry point
      figures/                          the numbered figures the report shows
notebooks/<slug>/                       one self-contained notebook per exercise
results/<slug>/                         generated tables and results.json quoted by the report
datasets/                               third-party data, git-ignored (see below)
briefs/                                 statement analyses used to plan an activity, not published
.github/workflows/deploy.yml            builds and publishes the site on push to main
```

`results/` sits outside `docs/` on purpose: MkDocs would otherwise publish every
generated table as its own page. The report pulls them in with the snippet
syntax — `--8<-- "results/<slug>/tbl_foo.md"`.

### Adding an activity

1. `docs/exercises/<slug>/` with `index.md` (front matter `exercise:` and `ai_use:`),
   `code.md`, `code/`, `figures/`.
2. `notebooks/<slug>/` and `results/<slug>/` if the activity needs them.
3. One more block under `nav:` in `mkdocs.yml`.
4. One more row in the table of `docs/index.md` and in the table above.

## Reproducing

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
mkdocs serve                 # preview at http://127.0.0.1:8000
```

Third-party datasets are **not** redistributed here. Activity 1 uses the Kaggle
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) `train.csv`
(8693 rows × 14 columns); download it and place it at
`datasets/spaceship-titanic/train.csv`. Then:

```bash
python docs/exercises/data/code/run_report.py   # regenerates docs/exercises/data/figures/* and results/data/*
python docs/exercises/perceptron/code/run_report.py  # regenerates Activity 2 figures and results
```

Pushing to `main` builds and publishes the site through GitHub Actions
(`.github/workflows/deploy.yml`).
