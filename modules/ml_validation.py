
"""
DataLens ML validation and splitting utilities.
"""

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold
)


def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=False
):
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if len(X) < 2:
        raise ValueError("At least two samples are required.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    stratify_values = None

    if stratify:
        class_counts = y.value_counts()

        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify_values = y

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )


def validate_target(y):
    if y is None:
        raise ValueError("Target variable is missing.")

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if len(y) < 2:
        raise ValueError("Target must contain at least two rows.")

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two unique values.")

    return True


def get_cross_validation_strategy(y, task="Regression", folds=5):
    if folds < 2:
        raise ValueError("At least two folds are required.")

    if task == "Classification":
        counts = y.value_counts()

        if counts.min() < folds:
            raise ValueError(
                "Each class must contain at least as many samples "
                "as the number of folds."
            )

        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )


def evaluate_cross_validation(
    model,
    X,
    y,
    task="Regression",
    scoring=None,
    folds=5
):
    strategy = get_cross_validation_strategy(
        y, task, folds
    )

    if scoring is None:
        scoring = (
            "accuracy" if task == "Classification"
            else "r2"
        )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=strategy,
        scoring=scoring
    )

    return {
        "scores": scores.tolist(),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std())
    }
