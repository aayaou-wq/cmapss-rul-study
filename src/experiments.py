import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from src.models import get_model


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_one_model(
    model_name,
    X_train_final,
    y_train_final,
    X_val,
    y_val,
    X_test,
    y_test,
    config
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["RANDOM_STATE"])

    input_shape = (X_train_final.shape[1], X_train_final.shape[2])

    model = get_model(model_name, input_shape, config)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["PATIENCE"],
            restore_best_weights=True
        )
    ]

    start_time = time.time()

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        verbose=config["VERBOSE"],
        callbacks=callbacks
    )

    elapsed = time.time() - start_time

    val_loss, val_rmse = model.evaluate(X_val, y_val, verbose=0)

    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    result = {
        "subset": config["SUBSET"],
        "model": model_name,
        "window_size": config["WINDOW_SIZE"],
        "rul_cap": config["RUL_CAP"],
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
        "epochs_ran": len(history.history["loss"]),
        "train_time_sec": float(elapsed),
        "n_train_samples": int(len(X_train_final)),
        "n_val_samples": int(len(X_val)),
        "n_test_samples": int(len(X_test)),
    }

    return result, history.history, model


def run_model_suite(
    model_names,
    X_train_final,
    y_train_final,
    X_val,
    y_val,
    X_test,
    y_test,
    config
):
    ensure_dir(config["OUTPUT_DIR"])

    results = []
    histories = {}

    for model_name in model_names:
        print(f"\nRunning: {model_name}")
        result, history, _ = run_one_model(
            model_name=model_name,
            X_train_final=X_train_final,
            y_train_final=y_train_final,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=config
        )
        results.append(result)
        histories[model_name] = history

    results_df = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)

    csv_path = os.path.join(config["OUTPUT_DIR"], f"results_{config['SUBSET']}.csv")
    json_path = os.path.join(config["OUTPUT_DIR"], f"results_{config['SUBSET']}.json")
    hist_path = os.path.join(config["OUTPUT_DIR"], f"histories_{config['SUBSET']}.json")

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(hist_path, "w") as f:
        json.dump(histories, f, indent=2)

    return results_df, histories
