
"""
DataLens ML-03: Classification Module

Provides classification model training, evaluation,
confusion matrix, ROC-AUC and model export.
"""

from io import BytesIO

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
    label_binarize
)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import joblib


# =========================================================
# CLASSIFICATION ALGORITHM REGISTRY
# =========================================================

def get_classification_models(random_state=42):

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state
        ),

        "KNN Classifier": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state
        ),

        "AdaBoost Classifier": AdaBoostClassifier(
            n_estimators=100,
            random_state=random_state
        ),

        "Hist Gradient Boosting Classifier":
            HistGradientBoostingClassifier(
                random_state=random_state
            ),

        "Support Vector Classifier": SVC(
            probability=True,
            random_state=random_state
        ),

        "Gaussian Naive Bayes": GaussianNB(),

        "Multinomial Naive Bayes": MultinomialNB(),

        "Bernoulli Naive Bayes": BernoulliNB(),

        "Linear Discriminant Analysis":
            LinearDiscriminantAnalysis(),

        "Quadratic Discriminant Analysis":
            QuadraticDiscriminantAnalysis()
    }


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_classification_data(df, target):

    if df is None or df.empty:
        raise ValueError("Please provide a non-empty dataset.")

    if target not in df.columns:
        raise ValueError("Selected target column does not exist.")

    working_df = df.copy()

    working_df = working_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    working_df = working_df.dropna(subset=[target])

    if len(working_df) < 5:
        raise ValueError(
            "At least 5 valid target observations are required."
        )

    X = working_df.drop(columns=[target]).copy()
    y = working_df[target].copy()

    if y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    # Remove completely empty features.
    usable_columns = X.columns[X.notna().any()].tolist()
    X = X[usable_columns]

    if X.empty:
        raise ValueError("No usable feature columns were found.")

    # Extract datetime features.
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in datetime_columns:

        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day
        X[f"{column}_dayofweek"] = X[column].dt.dayofweek

        X.drop(columns=[column], inplace=True)

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    supported_columns = (
        numerical_columns + categorical_columns
    )

    X = X[supported_columns]

    if X.empty:
        raise ValueError(
            "No supported numerical or categorical features found."
        )

    # Encode target labels.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    return (
        X,
        y_encoded,
        label_encoder,
        numerical_columns,
        categorical_columns
    )


# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

def build_classification_preprocessor(
    numerical_columns,
    categorical_columns
):

    transformers = []

    if numerical_columns:

        numerical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# MODEL TRAINING
# =========================================================

def train_classification_model(
    df,
    target,
    model_name,
    test_size=0.2,
    random_state=42
):

    (
        X,
        y,
        label_encoder,
        numerical_columns,
        categorical_columns
    ) = prepare_classification_data(df, target)

    if not 0 < test_size < 1:
        raise ValueError(
            "Test size must be between 0 and 1."
        )

    models = get_classification_models(random_state)

    if model_name not in models:
        raise ValueError(
            "Unsupported classification algorithm."
        )

    class_counts = pd.Series(y).value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Every target class must contain at least "
            "two observations for stratified splitting."
        )

    number_of_classes = len(class_counts)

    # Ensure every class can be represented in the test set.
    minimum_test_size = number_of_classes / len(y)

    if test_size < minimum_test_size:
        raise ValueError(
            f"Test size is too small for {number_of_classes} classes. "
            f"Increase it to at least "
            f"{minimum_test_size * 100:.1f}%."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    estimator = models[model_name]

    # KNN requires enough training observations.
    if model_name == "KNN Classifier":
        estimator.set_params(
            n_neighbors=min(5, len(X_train))
        )

    preprocessor = build_classification_preprocessor(
        numerical_columns,
        categorical_columns
    )

    # MultinomialNB requires non-negative input.
    # MinMax scaling is used for its numerical features.
    if model_name == "Multinomial Naive Bayes":

        from sklearn.preprocessing import MinMaxScaler

        transformers = []

        if numerical_columns:

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    MinMaxScaler()
                )
            ])

            transformers.append(
                (
                    "numerical",
                    numeric_pipeline,
                    numerical_columns
                )
            )

        if categorical_columns:

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ])

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_))
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    # ROC-AUC is available only for supported probability outputs.
    roc_auc = None

    try:

        if hasattr(pipeline, "predict_proba"):

            probabilities = pipeline.predict_proba(X_test)

            if number_of_classes == 2:

                roc_auc = roc_auc_score(
                    y_test,
                    probabilities[:, 1]
                )

            else:

                roc_auc = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=np.arange(number_of_classes)
                )

    except (ValueError, AttributeError):
        roc_auc = None

    results = {
        "model": pipeline,
        "model_name": model_name,
        "target": target,
        "metrics": {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1),
            "ROC-AUC": (
                float(roc_auc)
                if roc_auc is not None
                else None
            )
        },
        "confusion_matrix": matrix,
        "class_names": label_encoder.classes_.tolist(),
        "actual": label_encoder.inverse_transform(y_test),
        "predicted": label_encoder.inverse_transform(predictions),
        "classification_report": report,
        "label_encoder": label_encoder,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "feature_count": X.shape[1],
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "test_size": test_size
    }

    return results


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_classification_feature_importance(trained_pipeline):

    model = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]

    if not hasattr(preprocessor, "get_feature_names_out"):
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):

        importance_values = model.feature_importances_

    elif hasattr(model, "coef_"):

        importance_values = np.mean(
            np.abs(model.coef_),
            axis=0
        )

    else:
        return None

    if len(feature_names) != len(importance_values):
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df["Absolute Importance"] = (
        importance_df["Importance"].abs()
    )

    return importance_df.sort_values(
        "Absolute Importance",
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# MODEL EXPORT
# =========================================================

def export_classification_model(trained_pipeline):

    buffer = BytesIO()

    joblib.dump(trained_pipeline, buffer)

    buffer.seek(0)

    return buffer.getvalue()
