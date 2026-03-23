import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_lstm(input_shape, rnn_units=64, dense_units=64, dropout=0.2, learning_rate=1e-3):
    inp = layers.Input(shape=input_shape)

    x = layers.LSTM(rnn_units)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="lstm_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def build_gru(input_shape, rnn_units=64, dense_units=64, dropout=0.2, learning_rate=1e-3):
    inp = layers.Input(shape=input_shape)

    x = layers.GRU(rnn_units)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="gru_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def build_cnn_lstm(
    input_shape,
    conv_filters=64,
    kernel_size=3,
    rnn_units=64,
    dense_units=64,
    dropout=0.2,
    learning_rate=1e-3
):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu"
    )(inp)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(rnn_units)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="cnn_lstm_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def build_model(model_name, input_shape, config):
    if model_name == "lstm":
        return build_lstm(
            input_shape=input_shape,
            rnn_units=config["rnn_units"],
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"]
        )

    elif model_name == "gru":
        return build_gru(
            input_shape=input_shape,
            rnn_units=config["rnn_units"],
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"]
        )

    elif model_name == "cnn_lstm":
        return build_cnn_lstm(
            input_shape=input_shape,
            conv_filters=config["conv_filters"],
            kernel_size=config["kernel_size"],
            rnn_units=config["rnn_units"],
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"]
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
