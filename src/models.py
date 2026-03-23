from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


def build_lstm_model(
    input_shape,
    lstm_units=64,
    dropout_rate=0.2
):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    return model
