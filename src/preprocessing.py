"""Section 9 - scikit-learn preprocessing. Everything here is fit on TRAINING rows only."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOW_CARD = ["production_line", "shift"]
HIGH_CARD = ["supplier_id"]
NON_FEATURES = ["event_id", "production_timestamp", "label_avail_ts", "rejected", "label_ok", "split"]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its relative frequency in the TRAINING data.
    - Scales to high-cardinality columns (1 output column instead of one per level).
    - Unseen categories (e.g. a newly onboarded supplier) -> 0.0, i.e. 'never seen in training'.
    - Categories rarer than `min_freq` are pooled with unseen ones (0.0) so the model cannot memorise tiny groups."""
    def __init__(self, min_freq=0.005):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy(); self.columns_ = list(X.columns); self.maps_ = {}
        for c in self.columns_:
            f = X[c].astype(str).value_counts(normalize=True)
            self.maps_[c] = f[f >= self.min_freq].to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return np.column_stack([X[c].astype(str).map(self.maps_[c]).fillna(0.0).astype(float).values for c in self.columns_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])


def make_preprocessor(numeric_cols, scale=True):
    """Numeric: (optional) scale -> median impute + missing indicators (indicators only for columns with NaN in TRAIN).
    Low-cardinality categoricals: one-hot, unknown -> all zeros. High-cardinality: frequency encoding."""
    steps = ([("scale", StandardScaler())] if scale else []) + \
            [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    return ColumnTransformer([
        ("num", Pipeline(steps), numeric_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD),
        ("freq", FrequencyEncoder(), HIGH_CARD),
    ], verbose_feature_names_out=False)
