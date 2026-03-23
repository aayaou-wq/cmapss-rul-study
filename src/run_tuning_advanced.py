from .models_advanced import build_advanced_model
from .train_utils import set_seed, train_and_evaluate_model


def tune_cnn_lstm_attention(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed=42
):
    set_seed(seed)

    input_shape = (X_train.shape[1], X_train.shape[2])
    all_results = []
    trial_id = 0

    conv_filters_list = [32, 64, 128]
    kernel_size_list = [3, 5]
    rnn_units_list = [32, 64, 128]
    dense_units_list = [32, 64, 128]
    dropout_list = [0.1, 0.2, 0.3]
    learning_rate_list = [1e-3, 5e-4, 1e-4]
    num_heads_list = [2, 4]
    key_dim_list = [8, 16, 32]
    batch_size_list = [64, 128]

    for conv_filters in conv_filters_list:
        for kernel_size in kernel_size_list:
            for rnn_units in rnn_units_list:
                for dense_units in dense_units_list:
                    for dropout in dropout_list:
                        for learning_rate in learning_rate_list:
                            for num_heads in num_heads_list:
                                for key_dim in key_dim_list:
                                    for batch_size in batch_size_list:
                                        trial_id += 1

                                        config = {
                                            "conv_filters": conv_filters,
                                            "kernel_size": kernel_size,
                                            "rnn_units": rnn_units,
                                            "dense_units": dense_units,
                                            "dropout": dropout,
                                            "learning_rate": learning_rate,
                                            "num_heads": num_heads,
                                            "key_dim": key_dim
                                        }

                                        print("\n======================================")
                                        print(f"Trial {trial_id}")
                                        print(config)
                                        print(f"batch_size={batch_size}")
                                        print("======================================")

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
                                            epochs=15,
                                            batch_size=batch_size
                                        )

                                        results["trial"] = trial_id
                                        results["model"] = "cnn_lstm_att"
                                        results["conv_filters"] = conv_filters
                                        results["kernel_size"] = kernel_size
                                        results["rnn_units"] = rnn_units
                                        results["dense_units"] = dense_units
                                        results["dropout"] = dropout
                                        results["learning_rate"] = learning_rate
                                        results["num_heads"] = num_heads
                                        results["key_dim"] = key_dim
                                        results["batch_size"] = batch_size

                                        all_results.append(results)
                                        print(results)

    all_results = sorted(all_results, key=lambda x: x["val_rmse"])
    return all_results
