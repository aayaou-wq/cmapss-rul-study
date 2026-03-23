import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D


def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    return model


def build_gru_model(input_shape, gru_units=64, dropout_rate=0.2):
    model = Sequential([
        Input(shape=input_shape),
        GRU(gru_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    return model


def build_cnn_lstm_model(
    input_shape,
    filters=64,
    kernel_size=3,
    pool_size=2,
    lstm_units=64,
    dropout_rate=0.2
):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"),
        MaxPooling1D(pool_size=pool_size),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    return model


def get_model(model_name, input_shape, config):
    model_name = model_name.lower()

    if model_name == "lstm":
        return build_lstm_model(
            input_shape=input_shape,
            lstm_units=config["LSTM_UNITS"],
            dropout_rate=config["DROPOUT_RATE"]
        )

    if model_name == "gru":
        return build_gru_model(
            input_shape=input_shape,
            gru_units=config["GRU_UNITS"],
            dropout_rate=config["DROPOUT_RATE"]
        )

    if model_name == "cnn_lstm":
        return build_cnn_lstm_model(
            input_shape=input_shape,
            filters=config["CNN_FILTERS"],
            kernel_size=config["CNN_KERNEL_SIZE"],
            pool_size=config["POOL_SIZE"],
            lstm_units=config["LSTM_UNITS"],
            dropout_rate=config["DROPOUT_RATE"]
        )

    raise ValueError(f"Unsupported model_name: {model_name}")
