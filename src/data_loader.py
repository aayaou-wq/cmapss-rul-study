from pathlib import Path
import pandas as pd


FEATURE_COLUMNS = [
    "unit_nr", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
    "s20", "s21"
]


def _read_txt_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = FEATURE_COLUMNS
    return df


def _read_rul_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = ["RUL"]
    df["unit_nr"] = range(1, len(df) + 1)
    return df


def load_subset(data_dir: str, subset: str):
    """
    Load one C-MAPSS subset.

    Parameters
    ----------
    data_dir : str
        Folder containing train/test/RUL txt files.
    subset : str
        One of: FD001, FD002, FD003, FD004

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    rul_df : pd.DataFrame
    """
    subset = subset.upper()
    valid_subsets = {"FD001", "FD002", "FD003", "FD004"}
    if subset not in valid_subsets:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {sorted(valid_subsets)}")

    data_path = Path(data_dir)

    train_path = data_path / f"train_{subset}.txt"
    test_path = data_path / f"test_{subset}.txt"
    rul_path = data_path / f"RUL_{subset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing RUL file: {rul_path}")

    train_df = _read_txt_file(train_path)
    test_df = _read_txt_file(test_path)
    rul_df = _read_rul_file(rul_path)

    return train_df, test_df, rul_df
