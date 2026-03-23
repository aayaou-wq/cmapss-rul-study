import numpy as np
import pandas as pd


DEFAULT_TARGET_COLUMN = "RUL"
DEFAULT_UNIT_COLUMN = "unit_nr"
DEFAULT_TIME_COLUMN = "time_cycles"


def build_train_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int,
    target_col: str = DEFAULT_TARGET_COLUMN,
    unit_col: str = DEFAULT_UNIT_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
):
    X, y = [], []

    for unit_id in df[unit_col].unique():
        unit_df = df[df[unit_col] == unit_id].sort_values(time_col)

        features = unit_df[feature_cols].values
        targets = unit_df[target_col].values

        if len(unit_df) < window_size:
            continue

        for i in range(len(unit_df) - window_size + 1):
            X.append(features[i:i + window_size])
            y.append(targets[i + window_size - 1])

    return np.array(X), np.array(y)


def build_test_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int,
    target_col: str = DEFAULT_TARGET_COLUMN,
    unit_col: str = DEFAULT_UNIT_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
):
    X, y = [], []

    for unit_id in df[unit_col].unique():
        unit_df = df[df[unit_col] == unit_id].sort_values(time_col)

        features = unit_df[feature_cols].values
        targets = unit_df[target_col].values

        if len(unit_df) >= window_size:
            X.append(features[-window_size:])
        else:
            pad_len = window_size - len(unit_df)
            pad_block = np.repeat(features[:1], pad_len, axis=0)
            padded = np.vstack([pad_block, features])
            X.append(padded)

        y.append(targets[-1])

    return np.array(X), np.array(y)
