import time
import math
import random
import numpy as np
import tensorflow as tf

from dataclasses import dataclass, asdict
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, callbacks, optimizers


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ModelConfig:
    seed: int = 42
    model_name: str = "lstm"   # lstm, gru, cnn_lstm, lstm_att

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3

    rnn_units: int = 64
    conv_filters: int = 64
    kernel_size: int = 3
    dense_units: int = 64
    dropout: float = 0.2

    att_num_heads: int = 4
    att_key_dim: int = 16


# ============================================================
# METRICS
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    s = 0.0
    for x in d:
        if x < 0:
            s += math.exp(-x / 13.0) - 1.0
        else:
            s += math.exp(x / 10.0) - 1.0
    return float(s)


# ============================================================
# CALLBACKS
# ============================================================

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


# ============================================================
# MODELS
# ============================================================

def build_lstm(input_shape, cfg: ModelConfig):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(cfg.rnn_units)(inp)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="lstm_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse"
    )
    return model


def build_gru(input_shape, cfg: ModelConfig):
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(cfg.rnn_units)(inp)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="gru_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse"
    )
    return model


def build_cnn_lstm(input_shape, cfg: ModelConfig):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(
        filters=cfg.conv_filters,
        kernel_size=cfg.kernel_size,
        padding="same",
        activation="relu"
    )(inp)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.LSTM(cfg.rnn_units)(x)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="cnn_lstm_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse"
    )
    return model


def build_lstm_att(input_shape, cfg: ModelConfig):
    inp = layers.Input(shape=input_shape)

    x = layers.LSTM(cfg.rnn_units, return_sequences=True)(inp)
    x = layers.Dropout(cfg.dropout)(x)

    att = layers.MultiHeadAttention(
        num_heads=cfg.att_num_heads,
        key_dim=cfg.att_key_dim
    )(x, x)

    x = layers.Add()([x, att])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="lstm_attention_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse"
    )
    return model


def build_model(input_shape, cfg: ModelConfig):
    if cfg.model_name == "lstm":
        return build_lstm(input_shape, cfg)
    elif cfg.model_name == "gru":
        return build_gru(input_shape, cfg)
    elif cfg.model_name == "cnn_lstm":
        return build_cnn_lstm(input_shape, cfg)
    elif cfg.model_name == "lstm_att":
        return build_lstm_att(input_shape, cfg)
    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_and_evaluate_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    cfg: ModelConfig
):
    set_seed(cfg.seed)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, cfg)

    start = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=get_callbacks(),
        verbose=1
    )

    train_time = time.time() - start

    val_pred = model.predict(X_val, verbose=0).reshape(-1)
    test_pred = model.predict(X_test, verbose=0).reshape(-1)

    results = {
        "model": cfg.model_name,
        "val_rmse": round(rmse(y_val, val_pred), 4),
        "test_rmse": round(rmse(y_test, test_pred), 4),
        "test_score": round(nasa_score(y_test, test_pred), 4),
        "epochs_ran": len(history.history["loss"]),
        "train_time_sec": round(train_time, 2),
        "config": asdict(cfg)
    }

    return model, history, val_pred, test_pred, results


# ============================================================
# SIMPLE BASELINE RUNNER
# ============================================================

def run_baselines(X_train, y_train, X_val, y_val, X_test, y_test):
    all_results = []

    for model_name in ["lstm", "gru", "cnn_lstm"]:
        cfg = ModelConfig(
            model_name=model_name,
            epochs=10,
            batch_size=128,
            learning_rate=1e-3,
            rnn_units=64,
            conv_filters=64,
            kernel_size=3,
            dense_units=64,
            dropout=0.2
        )

        _, _, _, _, results = train_and_evaluate_model(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            cfg
        )

        all_results.append(results)
        print(results)

    return all_results


# ============================================================
# LIGHT DIAGNOSTIC FOR CNN-LSTM
# ============================================================

def search_cnn_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    all_results = []

    for conv_filters in [32, 64, 128]:
        for kernel_size in [3, 5]:
            for rnn_units in [32, 64, 128]:
                for dropout in [0.1, 0.2, 0.3]:
                    for learning_rate in [1e-3, 5e-4]:

                        cfg = ModelConfig(
                            model_name="cnn_lstm",
                            epochs=12,
                            batch_size=128,
                            learning_rate=learning_rate,
                            rnn_units=rnn_units,
                            conv_filters=conv_filters,
                            kernel_size=kernel_size,
                            dense_units=64,
                            dropout=dropout
                        )

                        _, _, _, _, results = train_and_evaluate_model(
                            X_train, y_train,
                            X_val, y_val,
                            X_test, y_test,
                            cfg
                        )

                        all_results.append(results)
                        print(results)

    all_results = sorted(all_results, key=lambda x: x["val_rmse"])
    return all_results


# ============================================================
# SEARCH FOR LSTM-ATT
# ============================================================

def search_lstm_att(X_train, y_train, X_val, y_val, X_test, y_test):
    all_results = []

    for rnn_units in [64, 128]:
        for att_num_heads in [2, 4]:
            for att_key_dim in [16, 32]:
                for dense_units in [64, 128]:
                    for dropout in [0.1, 0.2, 0.3]:
                        for learning_rate in [1e-3, 5e-4]:
                            for batch_size in [64, 128]:

                                cfg = ModelConfig(
                                    model_name="lstm_att",
                                    epochs=15,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    rnn_units=rnn_units,
                                    dense_units=dense_units,
                                    dropout=dropout,
                                    att_num_heads=att_num_heads,
                                    att_key_dim=att_key_dim
                                )

                                _, _, _, _, results = train_and_evaluate_model(
                                    X_train, y_train,
                                    X_val, y_val,
                                    X_test, y_test,
                                    cfg
                                )

                                all_results.append(results)
                                print(results)

    all_results = sorted(all_results, key=lambda x: x["val_rmse"])
    return all_results
