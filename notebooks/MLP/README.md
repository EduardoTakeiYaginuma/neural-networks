# MLP notebook

`mlp.ipynb` is the executed companion notebook for Activity 3. It follows the
statement in order and displays the complete NumPy MLP, every derivation,
Figures 1–6, analyses, and the final results table. It executes the canonical
files under `docs/exercises/mlp/code/` so the notebook and published report
cannot silently drift into different implementations.

From the repository root:

```bash
source env/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/MLP/mlp.ipynb
```
