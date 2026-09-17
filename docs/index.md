# Neural Networks — Reports

<p class="eyebrow">Insper · Artificial Neural Networks and Deep Learning</p>

Reports for the course activities. Each activity is a self-contained deliverable in the
[repository](https://github.com/EduardoTakeiYaginuma/neural-networks): the report is the
page you read here, its code lives beside it as runnable files, and a single seeded entry
point regenerates every figure and recomputes every number quoted in the text.

| # | Activity | Report | Code |
|---|---|---|---|
| 1 | Data preparation and analysis | [Report](exercises/data/index.md) | [Code](exercises/data/code.md) |
| 2 | Perceptron | [Report](exercises/perceptron/index.md) | [Code](exercises/perceptron/code.md) |

**Author:** Eduardo Takei Yaginuma

## Reproducing locally

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
mkdocs serve                  # preview the site at http://127.0.0.1:8000
```

Each activity has its own entry point:

```bash
python docs/exercises/data/code/run_report.py   # rewrites docs/exercises/data/figures/*.png and results/data/*
python docs/exercises/perceptron/code/run_report.py  # Activity 2 figures and results
```
