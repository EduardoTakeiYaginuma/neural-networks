# Neural Networks — Reports

<p class="eyebrow">Insper · Artificial Neural Networks and Deep Learning</p>

Reports for the course activities. Every report is generated from the code in the
[repository](https://github.com/EduardoTakeiYaginuma/neural-networks): a single
entry point (`python docs/exercises/data/code/run_report.py`) creates one seeded random generator,
regenerates every figure and recomputes every number quoted in the text.

| Activity | Report | Code |
|---|---|---|
| 1 — Data preparation and analysis | [Report](exercises/data/index.md) | [Code](exercises/data/code.md) |

**Author:** Eduardo Takei Yaginuma

## Reproducing locally

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python docs/exercises/data/code/run_report.py   # rewrites docs/exercises/data/figures/*.png and results/*
mkdocs serve                  # preview the site at http://127.0.0.1:8000
```
