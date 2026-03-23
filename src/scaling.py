import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_minmax_scaler(
    train_df: pd.DataFrame,
    feature_cols: list,
    feature_range: tuple = (-1, 1)
):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler,
    feature_cols: list
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out
