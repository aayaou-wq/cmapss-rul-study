import numpy as np
import pandas as pd

# =========================
# STATIC VARIABLES
# =========================
WINDOW_SIZE = 30
FEATURE_COLUMNS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
    "s20", "s21"
]
TARGET_COLUMN = "RUL"
UNIT_COL = "unit_nr"


def build_train_sequences(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: list = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    unit_col: str = UNIT_COL
):
    X, y = [], []

    for unit_id in df[unit_col].unique():
        unit_df = df[df[unit_col] == unit_id].sort_values("time_cycles")
        features = unit_df[feature_cols].values
        targets = unit_df[target_col].values

        for i in range(len(unit_df) - window_size + 1):
            X.append(features[i:i + window_size])
            y.append(targets[i + window_size - 1])

    return np.array(X), np.array(y)


def build_test_sequences(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: list = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    unit_col: str = UNIT_COL
):
    X, y = [], []

    for unit_id in df[unit_col].unique():
        unit_df = df[df[unit_col] == unit_id].sort_values("time_cycles")
        features = unit_df[feature_cols].values
        targets = unit_df[target_col].values

        if len(unit_df) >= window_size:
            X.append(features[-window_size:])
            y.append(targets[-1])
        else:
            pad_len = window_size - len(unit_df)
            pad_block = np.repeat(features[:1], pad_len, axis=0)
            padded_features = np.vstack([pad_block, features])
            X.append(padded_features)
            y.append(targets[-1])

    return np.array(X), np.array(y)
