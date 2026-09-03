"""
Single entry point for the whole report.

One Generator (np.random.default_rng(42)) is created here and threaded through
every exercise, so "the same rng throughout the report" is literally true and
re-running this file reproduces every figure and every number.

    python src/run_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
import ex1_point_clouds as ex1
import ex2_nonlinearity as ex2
import ex3_realworld as ex3

RESULTS = style.ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def md_table(rows, header):
    """Minimal GitHub-flavoured markdown table writer."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write(name, text):
    (RESULTS / name).write_text(text)
    print(f"  table  -> results/{name}")


def main():
    rng = np.random.default_rng(42)          # the one and only seed

    r1 = ex1.run(rng)
    r2 = ex2.run(rng)
    r3 = ex3.run()

    # ---------------------------------------------------------------- Exercise 1
    smallest = r1["smallest_r_s1"]["pair"]
    write("tbl_ex1_separation.md", md_table(
        [[p["pair"],
          f'{p["distance"]:.4f}',
          f'{p["sigma_sum"]:.4f}',
          (f'**{p["r"]:.4f}** &larr; smallest' if p["pair"] == smallest else f'{p["r"]:.4f}'),
          f'{p["r"] / 2.0:.4f}']
         for p in r1["separation_ratios_s1"]],
        ["Pair (i, j)", "‖μᵢ − μⱼ‖",
         "σ̄ᵢ + σ̄ⱼ (s = 1)", "rᵢⱼ at s = 1", "rᵢⱼ at s = 2"]))

    write("tbl_ex1_mixing.md", md_table(
        [[f"{float(s):.1f}", f"{v:.2%}", f"{int(round(v * 400))} of 400"]
         for s, v in r1["mixing_rates"].items()],
        ["Scale factor s", "Mixing rate", "Misplaced points"]))

    # ---------------------------------------------------------------- Exercise 2
    write("tbl_ex2_measures.md", md_table(
        [["Distance between class centres (5D)",
          f'{r2["dataset1"]["centre_distance"]:.4f}',
          f'{r2["dataset2"]["centre_distance"]:.4f}'],
         ["Explained variance PC1",
          f'{r2["dataset1"]["evr"][0]:.2%}', f'{r2["dataset2"]["evr"][0]:.2%}'],
         ["Explained variance PC2",
          f'{r2["dataset1"]["evr"][1]:.2%}', f'{r2["dataset2"]["evr"][1]:.2%}'],
         ["Explained variance PC1 + PC2",
          f'**{r2["dataset1"]["evr_sum"]:.2%}**', f'**{r2["dataset2"]["evr_sum"]:.2%}**']],
        ["Measure", "Dataset I (Gaussians)", "Dataset II (shells)"]))

    rc, rd = r2["dataset2"]["radius_C"], r2["dataset2"]["radius_D"]
    write("tbl_ex2_radii.md", md_table(
        [["Class C (core)", f"{rc[0]:.4f}", f"{rc[1]:.4f}", f"{rc[2]:.4f}"],
         ["Class D (shell)", f"{rd[0]:.4f}", f"{rd[1]:.4f}", f"{rd[2]:.4f}"]],
        ["Class", "min ‖x‖", "mean ‖x‖", "max ‖x‖"]))

    # ---------------------------------------------------------------- Exercise 3
    write("tbl_ex3_missing.md", md_table(
        [[c, v["missing_count"], f'{v["missing_pct"]:.2f}%']
         for c, v in r3["missing"].items()],
        ["Column", "Missing (count)", "Missing (%)"]))

    write("tbl_ex3_spending.md", md_table(
        [[c, f'{v["mean"]:,.2f}', f'{v["median"]:,.2f}', f'{v["max"]:,.0f}']
         for c, v in r3["spend_stats"].items()],
        ["Column", "Mean", "Median", "Maximum"]))

    write("tbl_ex3_ranges.md", md_table(
        [[c, f'{v["train_min"]:.4f}', f'{v["train_max"]:.4f}',
          f'{v["test_min"]:.4f}', f'{v["test_max"]:.4f}']
         for c, v in r3["ranges"].items()],
        ["Feature", "train min", "train max", "test min", "test max"]))

    # ------------------------------------------------------------ final summary
    ck = r3["checks"]
    mr = r1["mixing_rates"]
    rows = [[i + 1, f"Mixing rate at s = {float(s_):.1f}", f'`{v:.2%}`']
            for i, (s_, v) in enumerate(mr.items())]
    write("tbl_summary.md", md_table(rows + [
        [5, "Smallest rᵢⱼ at s = 1.0, and which pair",
            f'`{r1["smallest_r_s1"]["r"]:.4f}` — pair {smallest}, i.e. classes 0 and 1'],
        [6, "Distance between centres — Dataset I", f'`{r2["dataset1"]["centre_distance"]:.4f}`'],
        [7, "Distance between centres — Dataset II", f'`{r2["dataset2"]["centre_distance"]:.4f}`'],
        [8, "Explained variance PC1 + PC2 — Dataset I", f'`{r2["dataset1"]["evr_sum"]:.2%}`'],
        [9, "Explained variance PC1 + PC2 — Dataset II", f'`{r2["dataset2"]["evr_sum"]:.2%}`'],
        [10, "Share of the positive class in `Transported`",
             f'`{r3["target_balance"]["True"]:.2%}` (True) vs `{r3["target_balance"]["False"]:.2%}` (False)'],
        [11, "Mean and median of `FoodCourt` on the training set, before transforming",
             f'mean `{r3["foodcourt_train_raw"]["mean"]:.2f}`, median `{r3["foodcourt_train_raw"]["median"]:.2f}`'],
        [12, "Final shape of the training feature matrix",
             f'`{tuple(ck["shape_train"])}`'],
        [13, "Minimum and maximum of the training and test sets after scaling",
             f'train `[{ck["train_min"]:.4f}, {ck["train_max"]:.4f}]`, '
             f'test `[{ck["test_min"]:.4f}, {ck["test_max"]:.4f}]`'],
    ], ["#", "Item", "Value"]))

    payload = {"exercise1": r1, "exercise2": r2, "exercise3": r3}
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2))
    print("  json   -> results/results.json")


if __name__ == "__main__":
    main()
