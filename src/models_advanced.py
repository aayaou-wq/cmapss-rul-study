import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_cnn_lstm_attention(
    input_shape,
    conv_filters=64,
    kernel_size=3,
    rnn_units=64,
    dense_units=64,
    dropout=0.2,
    learning_rate=1e-3,
    num_heads=4,
    key_dim=16
):
    inp = layers.Input(shape=input_shape)

    # CNN block
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu"
    )(inp)
    x = layers.Dropout(dropout)(x)

    # LSTM block
    x = layers.LSTM(rnn_units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)

    # Attention block
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x, x)

    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    # Prediction head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out, name="cnn_lstm_attention_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def build_advanced_model(model_name, input_shape, config):
    if model_name == "cnn_lstm_att":
        return build_cnn_lstm_attention(
            input_shape=input_shape,
            conv_filters=config["conv_filters"],
            kernel_size=config["kernel_size"],
            rnn_units=config["rnn_units"],
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            num_heads=config["num_heads"],
            key_dim=config["key_dim"]
        )

    raise ValueError(f"Unknown advanced model: {model_name}")
