from .models_baselines import build_model
from .train_utils import set_seed, train_and_evaluate_model


def run_all_baselines(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed=42
):
    set_seed(seed)

    input_shape = (X_train.shape[1], X_train.shape[2])

    base_config = {
        "rnn_units": 64,
        "conv_filters": 64,
        "kernel_size": 3,
        "dense_units": 64,
        "dropout": 0.2,
        "learning_rate": 1e-3
    }

    model_names = ["lstm", "gru", "cnn_lstm"]
    all_results = []

    for model_name in model_names:
        print(f"\n==============================")
        print(f"Running baseline: {model_name}")
        print(f"==============================")

        model = build_model(
            model_name=model_name,
            input_shape=input_shape,
            config=base_config
        )

        _, _, _, results = train_and_evaluate_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            epochs=10,
            batch_size=128
        )

        results["model"] = model_name
        all_results.append(results)

        print(results)

    all_results = sorted(all_results, key=lambda x: x["val_rmse"])
    return all_results
