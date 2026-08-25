"""
Exercise 3 - Preparing the Spaceship Titanic data for a tanh network.

The pipeline is deliberately explicit (no ColumnTransformer black box) so that
every statistic can be shown to come from the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import style
from style import SERIES, ACCENT, INK, INK_SOFT, MONO
import matplotlib.pyplot as plt

# --8<-- [start:columns]
CSV = style.ROOT / "data" / "train.csv"

TARGET = "Transported"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age"] + SPEND
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["Cabin", "Name", "PassengerId"]     # identifiers / high-cardinality text
SEED = 42
# --8<-- [end:columns]


def describe(df):
    """Item A - target balance, feature inventory, missing values, spending stats."""
    n = len(df)
    balance = df[TARGET].value_counts(normalize=True).sort_index()

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    }).sort_values("missing_count", ascending=False)

    spend_stats = df[SPEND].agg(["mean", "median", "max"]).T
    spend_stats["mean_over_median"] = np.where(
        spend_stats["median"] > 0, spend_stats["mean"] / spend_stats["median"], np.inf)

    print("Exercise 3 - Spaceship Titanic")
    print(f"  rows = {n}, columns = {df.shape[1]}")
    print(f"  Transported: False = {balance.get(False, 0):.4%}, True = {balance.get(True, 0):.4%}")
    print(f"  total missing cells = {int(df.isna().sum().sum())}")
    print(spend_stats.to_string())
    return balance, missing, spend_stats


# --8<-- [start:split]
def split(df):
    """80/20 stratified split with a fixed seed - before any statistic is computed."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
# --8<-- [end:split]


# --8<-- [start:preprocess]
def preprocess(X_train, X_test):
    """Fit every transformation on the training split, apply it to both.

    Steps, in order:
      1. drop identifier columns;
      2. impute numeric columns with the TRAINING median (robust to the heavy
         right tail of the spending columns, where the median is 0);
      3. impute categorical columns with the TRAINING most frequent category;
      4. engineer TotalSpend = sum of the five spending columns;
      5. log1p the spending columns (and TotalSpend) to compress the tails;
      6. one-hot encode the categorical columns, ignoring unseen categories;
      7. Min-Max scale every numeric column to [-1, 1], the range tanh lives in.
    """
    Xtr = X_train.drop(columns=DROP).copy()
    Xte = X_test.drop(columns=DROP).copy()

    # -- 2. numeric imputation (median learned on train only)
    num_imp = SimpleImputer(strategy="median").fit(Xtr[NUMERIC])
    Xtr[NUMERIC] = num_imp.transform(Xtr[NUMERIC])
    Xte[NUMERIC] = num_imp.transform(Xte[NUMERIC])

    # -- 3. categorical imputation (mode learned on train only); booleans as strings
    for d in (Xtr, Xte):
        d[CATEGORICAL] = d[CATEGORICAL].astype("object")
    cat_imp = SimpleImputer(strategy="most_frequent").fit(Xtr[CATEGORICAL])
    Xtr[CATEGORICAL] = cat_imp.transform(Xtr[CATEGORICAL])
    Xte[CATEGORICAL] = cat_imp.transform(Xte[CATEGORICAL])

    # -- 4. feature engineering
    Xtr["TotalSpend"] = Xtr[SPEND].sum(axis=1)
    Xte["TotalSpend"] = Xte[SPEND].sum(axis=1)

    raw_foodcourt_train = Xtr["FoodCourt"].copy()   # kept for Figure 6

    # -- 5. log1p on the heavy-tailed columns (a fixed function: no fitting, no leakage)
    heavy = SPEND + ["TotalSpend"]
    Xtr[heavy] = np.log1p(Xtr[heavy])
    Xte[heavy] = np.log1p(Xte[heavy])

    # -- 6. one-hot encoding; handle_unknown="ignore" -> unseen category becomes all zeros
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xtr[CATEGORICAL])
    ohe_cols = list(ohe.get_feature_names_out(CATEGORICAL))
    Otr = pd.DataFrame(ohe.transform(Xtr[CATEGORICAL]), columns=ohe_cols, index=Xtr.index)
    Ote = pd.DataFrame(ohe.transform(Xte[CATEGORICAL]), columns=ohe_cols, index=Xte.index)

    # -- 7. Min-Max scaling to [-1, 1] (range fitted on train only)
    num_cols = NUMERIC + ["TotalSpend"]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Xtr[num_cols])
    Ntr = pd.DataFrame(scaler.transform(Xtr[num_cols]), columns=num_cols, index=Xtr.index)
    Nte = pd.DataFrame(scaler.transform(Xte[num_cols]), columns=num_cols, index=Xte.index)

    # one-hot columns are already 0/1, i.e. inside [-1, 1]: they are not rescaled
    Ftr = pd.concat([Ntr, Otr], axis=1)
    Fte = pd.concat([Nte, Ote], axis=1)
    return Ftr, Fte, ohe_cols, raw_foodcourt_train, num_cols
# --8<-- [end:preprocess]


def figure6(raw, transformed):
    """Figure 6 - FoodCourt before and after log1p + Min-Max scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    axes[0].hist(raw, bins=60, color=SERIES[0], alpha=0.85,
                 edgecolor=style.SURFACE, linewidth=0.5, label="FoodCourt (raw)")
    axes[0].set_xlabel("FoodCourt (credits spent)")
    axes[0].set_ylabel("Passengers (training set)")
    axes[0].set_yscale("log")
    axes[0].set_title("Before - raw values, heavy right tail",
                      fontfamily=style.SERIF, fontsize=10.5, loc="left")
    axes[0].legend(loc="upper right")

    axes[1].hist(transformed, bins=60, color=SERIES[1], alpha=0.85,
                 edgecolor=style.SURFACE, linewidth=0.5,
                 label="FoodCourt (log1p + Min-Max)")
    axes[1].set_xlabel("FoodCourt after $\\log(1+x)$ and scaling to $[-1, 1]$")
    axes[1].set_ylabel("Passengers (training set)")
    axes[1].set_yscale("log")
    axes[1].axvline(-1, color=ACCENT, lw=1.0, ls="--")
    axes[1].axvline(1, color=ACCENT, lw=1.0, ls="--")
    axes[1].set_title("After - inside the tanh range $[-1, 1]$",
                      fontfamily=style.SERIF, fontsize=10.5, loc="left")
    axes[1].legend(loc="upper right")

    for ax in axes:
        style.mono_ticks(ax)
    fig.suptitle("Figure 6 - Effect of the heavy-tail transformation on FoodCourt (training set)",
                 fontfamily=style.SERIF, fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.tight_layout()
    return style.save(fig, "fig6_foodcourt.png")


def run(rng=None):
    df = pd.read_csv(CSV)
    balance, missing, spend_stats = describe(df)

    X_train, X_test, y_train, y_test = split(df)
    print(f"  split: train {X_train.shape[0]} rows, test {X_test.shape[0]} rows "
          f"(train positive share {y_train.mean():.4%}, test {y_test.mean():.4%})")

    fc_mean = float(X_train["FoodCourt"].mean())
    fc_median = float(X_train["FoodCourt"].median())
    print(f"  FoodCourt on train before transforming: mean = {fc_mean:.4f}, median = {fc_median:.4f}")

    Ftr, Fte, ohe_cols, raw_fc, num_cols = preprocess(X_train, X_test)

    figure6(raw_fc, Ftr["FoodCourt"])

    checks = {
        "nan_train": int(Ftr.isna().sum().sum()),
        "nan_test": int(Fte.isna().sum().sum()),
        "shape_train": list(Ftr.shape),
        "shape_test": list(Fte.shape),
        "train_min": float(Ftr.min().min()),
        "train_max": float(Ftr.max().max()),
        "test_min": float(Fte.min().min()),
        "test_max": float(Fte.max().max()),
    }
    print(f"  NaN after preprocessing: train {checks['nan_train']}, test {checks['nan_test']}")
    print(f"  final shapes: train {tuple(Ftr.shape)}, test {tuple(Fte.shape)}")
    print(f"  value range: train [{checks['train_min']:.4f}, {checks['train_max']:.4f}], "
          f"test [{checks['test_min']:.4f}, {checks['test_max']:.4f}]")
    print(f"  one-hot columns ({len(ohe_cols)}): {ohe_cols}")

    # per-column ranges, used in the report's verification table
    ranges = pd.DataFrame({"train_min": Ftr.min(), "train_max": Ftr.max(),
                           "test_min": Fte.min(), "test_max": Fte.max()}).round(4)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "target_balance": {"False": float(balance.get(False, 0)), "True": float(balance.get(True, 0))},
        "missing": missing.to_dict(orient="index"),
        "spend_stats": spend_stats.round(4).to_dict(orient="index"),
        "split": {"train_rows": int(X_train.shape[0]), "test_rows": int(X_test.shape[0]),
                  "train_pos": float(y_train.mean()), "test_pos": float(y_test.mean())},
        "foodcourt_train_raw": {"mean": fc_mean, "median": fc_median},
        "one_hot_columns": ohe_cols,
        "feature_columns": list(Ftr.columns),
        "checks": checks,
        "ranges": ranges.to_dict(orient="index"),
    }
