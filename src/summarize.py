import pandas as pd


def format_results_table(results_df: pd.DataFrame):
    df = results_df.copy()
    df["val_rmse"] = df["val_rmse"].round(4)
    df["test_rmse"] = df["test_rmse"].round(4)
    df["train_time_sec"] = df["train_time_sec"].round(2)

    return df[[
        "subset",
        "model",
        "window_size",
        "rul_cap",
        "val_rmse",
        "test_rmse",
        "epochs_ran",
        "train_time_sec",
        "n_train_samples",
        "n_val_samples",
        "n_test_samples"
    ]]
