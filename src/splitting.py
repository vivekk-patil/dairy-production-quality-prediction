"""Chronological 70/15/15 split with label-delay embargo."""
import numpy as np, pandas as pd


def chrono_split(F: pd.DataFrame, train=0.70, val=0.15):
    """F must be sorted by production_timestamp. Returns a Series with
    train / val / test / embargo / unlabeled.

    Embargo: a row from period P whose lab result is reported AFTER period P ends would not yet
    be a known label when the next period starts, so its label cannot be used for training/early
    stopping/selection at that point. Those rows (~last 24h of train and val) are set aside.
    """
    n = len(F); i1, i2 = int(n * train), int(n * (train + val))
    pos = np.arange(n)
    split = np.where(pos < i1, "train", np.where(pos < i2, "val", "test")).astype(object)
    t1 = F.production_timestamp.iloc[i1]; t2 = F.production_timestamp.iloc[i2]
    split[(split == "train") & (F.label_avail_ts.values > np.datetime64(t1))] = "embargo"
    split[(split == "val") & (F.label_avail_ts.values > np.datetime64(t2))] = "embargo"
    split[~F.label_ok.values] = np.where(split[~F.label_ok.values] == "test", "unlabeled", "unlabeled")
    return pd.Series(split, index=F.index, name="split"), (t1, t2)
