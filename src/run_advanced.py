from .models_advanced import build_advanced_model
from .train_utils import set_seed, train_and_evaluate_model


def run_cnn_lstm_attention(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed=42
):
    set_seed(seed)

    input_shape = (X_train.shape[1], X_train.shape[2])

    config = {
        "conv_filters": 64,
        "kernel_size": 3,
        "rnn_units": 64,
        "dense_units": 64,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "num_heads": 4,
        "key_dim": 16
    }

    model = build_advanced_model(
        model_name="cnn_lstm_att",
        input_shape=input_shape,
        config=config
    )

    _, _, _, results = train_and_evaluate_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        epochs=10,
        batch_size=128
    )

    results["model"] = "cnn_lstm_att"
    return results
