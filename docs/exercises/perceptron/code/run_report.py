"""Regenerate every figure, table and raw result for Activity 2.

Run from the repository root with:

    python docs/exercises/perceptron/code/run_report.py

Exactly one ``np.random.default_rng(42)`` is created here and threaded through
both exercises in statement order.
"""
from dataclasses import asdict
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
import ex1_separable as ex1
import ex2_overlapping as ex2

RESULTS = style.ROOT / "results" / "perceptron"
RESULTS.mkdir(parents=True, exist_ok=True)


def markdown_table(rows, header):
    """Return a small GitHub-flavoured Markdown table."""

    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |"
                 for row in rows)
    return "\n".join(lines) + "\n"


def write(name, contents):
    output = RESULTS / name
    output.write_text(contents)
    print(f"  table  -> {output.relative_to(style.ROOT)}")


def serializable(value):
    """Recursively convert dataclasses and NumPy values for JSON output."""

    if hasattr(value, "__dataclass_fields__"):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compact_result(r1, r2):
    """Keep reported metrics but omit the 4,000 raw points from results.json."""

    ex1_low = r1["eta_001"]
    ex1_high = r1["eta_1"]
    ex2_fit = r2["training"]
    return {
        "seed": 42,
        "exercise1": {
            "initial_weights": r1["initial_weights"],
            "eta_0.01": ex1_low,
            "eta_1.0": ex1_high,
            "direction_eta_0.01": r1["direction_eta_001"],
            "direction_eta_1.0": r1["direction_eta_1"],
            "direction_cosine": r1["direction_cosine"],
            "direction_angle_degrees": r1["direction_angle_degrees"],
        },
        "exercise2": {
            "initial_weights": r2["initial_weights"],
            "training": ex2_fit,
            "diagnostics": r2["diagnostics"],
        },
    }


def main():
    rng = np.random.default_rng(42)
    r1 = ex1.run(rng)
    r2 = ex2.run(rng)

    low = r1["eta_001"]
    high = r1["eta_1"]
    overlap = r2["training"]
    final_overlap_accuracy = r2["diagnostics"]["final_accuracy"]

    rows = [
        [1, r"Exercise 1 — final $\mathbf{w}$ and $b$",
         f"$\\mathbf{{w}}=[{low.weights[0]:.6f}, {low.weights[1]:.6f}]$, "
         f"$b={low.bias:.6f}$"],
        [2, "Exercise 1 — epochs to convergence", str(low.epochs)],
        [3, "Exercise 1 — final accuracy", f"{low.accuracy_history[-1]:.2%}"],
        [4, r"Exercise 1 — epochs and final accuracy with $\eta = 1.0$",
         f"{high.epochs} epochs; {high.accuracy_history[-1]:.2%}"],
        [5, r"Exercise 2 — final $\mathbf{w}$ and $b$",
         f"$\\mathbf{{w}}=[{overlap.weights[0]:.6f}, "
         f"{overlap.weights[1]:.6f}]$, $b={overlap.bias:.6f}$"],
        [6, "Exercise 2 — accuracy of the final weights",
         f"{final_overlap_accuracy:.2%}"],
        [7, "Exercise 2 — accuracy of the pocket weights",
         f"{overlap.pocket_accuracy:.2%}"],
        [8, "Exercise 2 — epoch at which the pocket best occurred",
         str(overlap.pocket_epoch)],
    ]
    write("tbl_summary.md", markdown_table(rows, ["#", "Quantity", "Value"]))

    payload = serializable(compact_result(r1, r2))
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  json   -> {(RESULTS / 'results.json').relative_to(style.ROOT)}")


if __name__ == "__main__":
    main()
