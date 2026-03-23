import time
import math
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from tensorflow.keras import callbacks


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    d = y_pred - y_true
    score = 0.0

    for di in d:
        if di < 0:
            score += math.exp(-di / 13.0) - 1.0
        else:
            score += math.exp(di / 10.0) - 1.0

    return float(score)


def get_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]


def train_and_evaluate_model(
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    epochs=10,
    batch_size=128
):
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=1
    )

    train_time_sec = time.time() - start_time

    val_pred = model.predict(X_val, verbose=0).reshape(-1)
    test_pred = model.predict(X_test, verbose=0).reshape(-1)

    results = {
        "val_rmse": round(rmse(y_val, val_pred), 4),
        "test_rmse": round(rmse(y_test, test_pred), 4),
        "test_score": round(nasa_score(y_test, test_pred), 4),
        "epochs_ran": len(history.history["loss"]),
        "train_time_sec": round(train_time_sec, 2),
    }

    return history, val_pred, test_pred, results
