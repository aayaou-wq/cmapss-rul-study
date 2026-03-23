from sklearn.model_selection import train_test_split


def split_train_validation(
    X,
    y,
    val_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True
):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        shuffle=shuffle
    )
    return X_train, X_val, y_train, y_val
