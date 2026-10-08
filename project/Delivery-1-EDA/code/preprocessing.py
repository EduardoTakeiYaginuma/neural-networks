"""Leakage-safe preprocessing used by the stroke-prediction project.

The notebook imports this module so that the exact fitted preprocessing can be
reused by the modeling deliverable.  No transformer is fitted at import time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric feature to Tukey fences learned during ``fit``.

    This is a winsorization-style treatment: it retains observations while
    limiting the leverage of extreme values.  It accepts either a DataFrame or
    an array and exposes feature names for ``ColumnTransformer``.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Return the unfitted preprocessing graph used throughout the project."""

    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outliers", IQRClipper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
