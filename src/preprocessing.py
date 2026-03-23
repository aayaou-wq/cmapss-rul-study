import pandas as pd


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_nr", "max_cycle"]
    df = df.merge(max_cycles, on="unit_nr", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df


def add_rul_to_test(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()

    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_nr", "max_cycle"]

    df = df.merge(max_cycles, on="unit_nr", how="left")
    df = df.merge(rul_df, on="unit_nr", how="left")
    df["RUL"] = (df["max_cycle"] - df["time_cycles"]) + df["RUL"]
    df = df.drop(columns=["max_cycle"])
    return df


def cap_rul(df: pd.DataFrame, cap_value: int = 125) -> pd.DataFrame:
    out = df.copy()
    out["RUL"] = out["RUL"].clip(upper=cap_value)
    return out
