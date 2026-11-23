
"""
DataLens Machine Learning Studio

ML-01: Foundation
ML-02: Regression
ML-03: Classification
ML-04: Clustering
ML-05: Dimensionality Reduction
ML-06: Anomaly Detection
ML-07: Association Rule Mining
"""

import streamlit as st
import pandas as pd

from modules.ml_registry import (
    get_algorithm_families,
    get_algorithms,
    get_algorithm_count
)

from modules.ml_preprocessing import (
    inspect_dataset,
    get_dataset_readiness,
    suggest_ml_task
)

from modules.ml_regression import (
    get_regression_models,
    train_regression_model,
    get_regression_feature_importance,
    export_regression_model
)

from modules.ml_classification import (
    get_classification_models,
    train_classification_model,
    get_classification_feature_importance,
    export_classification_model
)

from modules.ml_clustering import (
    get_clustering_models,
    train_clustering_model,
    get_cluster_summary,
    export_clustering_model,
    export_clustered_dataset
)

from modules.ml_dimensionality import (
    get_dimensionality_algorithms,
    run_dimensionality_reduction,
    export_dimensionality_model,
    export_reduced_dataset,
    get_component_summary
)

from modules.ml_anomaly import (
    get_anomaly_algorithms,
    run_anomaly_detection,
    export_anomaly_model,
    export_anomaly_dataset,
    export_anomaly_summary
)

from modules.ml_association import (
    get_association_algorithms,
    run_association_mining,
    filter_association_rules,
    export_frequent_itemsets,
    export_association_rules,
    export_association_model
)


# =========================================================
# SHARED HELPERS
# =========================================================

def _get_dataset():

    df = st.session_state.get("dataset")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("Please upload a non-empty dataset first.")
        return None

    return df


def _show_model_download(data, filename, key):

    st.download_button(
        "Download Trained Model (.joblib)",
        data=data,
        file_name=filename,
        mime="application/octet-stream",
        use_container_width=True,
        key=key
    )


# =========================================================
# REGRESSION
# =========================================================

def render_regression():

    st.subheader("Regression Analysis")
    st.write("Predict continuous numerical values using regression algorithms.")

    df = _get_dataset()
    if df is None:
        return

    targets = df.select_dtypes(include="number").columns.tolist()

    if not targets:
        st.warning("No numerical target columns are available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(
            "Prediction Target",
            targets,
            key="ml_regression_target"
        )

    with col2:
        model_name = st.selectbox(
            "Algorithm",
            list(get_regression_models().keys()),
            key="ml_regression_algorithm"
        )

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test Data (%)",
            10, 40, 20, 5,
            key="ml_regression_test_size"
        )

    with col2:
        random_state = st.number_input(
            "Random State",
            0, 10000, 42,
            key="ml_regression_random_state"
        )

    if st.button(
        "Train Regression Model",
        type="primary",
        use_container_width=True,
        key="ml_train_regression"
    ):
        try:
            with st.spinner("Training regression model..."):

                results = train_regression_model(
                    df=df,
                    target=target,
                    model_name=model_name,
                    test_size=test_size / 100,
                    random_state=int(random_state)
                )

                st.session_state["ml_regression_results"] = results
                st.session_state["ml_regression_config"] = (
                    target, model_name, test_size, int(random_state)
                )

            st.success("Regression model trained successfully.")

        except Exception as error:
            st.error(f"Regression training failed: {error}")

    results = st.session_state.get("ml_regression_results")

    if results is None:
        return

    config = (target, model_name, test_size, int(random_state))

    if config != st.session_state.get("ml_regression_config"):
        st.info("Configuration changed. Train the model again.")
        return

    metrics = results["metrics"]

    st.divider()
    st.markdown("### Model Evaluation")

    cols = st.columns(4)

    for col, name, key in zip(
        cols,
        ["MAE", "MSE", "RMSE", "R² Score"],
        ["MAE", "MSE", "RMSE", "R2"]
    ):
        col.metric(name, f"{metrics[key]:.4f}")

    st.markdown("### Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual": results["actual"].to_numpy(),
        "Predicted": results["predicted"].to_numpy()
    })

    st.scatter_chart(
        comparison,
        x="Actual",
        y="Predicted",
        use_container_width=True
    )

    with st.expander("Prediction Comparison Table"):
        st.dataframe(comparison, use_container_width=True)

    st.markdown("### Residual Analysis")

    residuals = pd.DataFrame({
        "Prediction": results["predicted"].to_numpy(),
        "Residual": (
            results["actual"].to_numpy()
            - results["predicted"].to_numpy()
        )
    })

    st.scatter_chart(
        residuals,
        x="Prediction",
        y="Residual",
        use_container_width=True
    )

    importance = get_regression_feature_importance(results["model"])

    if importance is not None and not importance.empty:
        st.markdown("### Feature Importance")
        st.bar_chart(
            importance.head(20).set_index("Feature")["Importance"],
            use_container_width=True
        )

    _show_model_download(
        export_regression_model(results["model"]),
        "datalens_regression_model.joblib",
        "ml_download_regression_model"
    )


# =========================================================
# CLASSIFICATION
# =========================================================

def render_classification():

    st.subheader("Classification Analysis")
    st.write("Predict categorical outcomes using classification algorithms.")

    df = _get_dataset()
    if df is None:
        return

    target = st.selectbox(
        "Classification Target",
        df.columns.tolist(),
        key="ml_classification_target"
    )

    class_count = df[target].dropna().nunique()

    st.caption(f"Target classes: {class_count}")

    if class_count < 2:
        st.warning("The selected target must contain at least two classes.")
        return

    model_name = st.selectbox(
        "Algorithm",
        list(get_classification_models().keys()),
        key="ml_classification_algorithm"
    )

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test Data (%)",
            10, 40, 20, 5,
            key="ml_classification_test_size"
        )

    with col2:
        random_state = st.number_input(
            "Random State",
            0, 10000, 42,
            key="ml_classification_random_state"
        )

    if st.button(
        "Train Classification Model",
        type="primary",
        use_container_width=True,
        key="ml_train_classification"
    ):
        try:
            with st.spinner("Training classification model..."):

                results = train_classification_model(
                    df=df,
                    target=target,
                    model_name=model_name,
                    test_size=test_size / 100,
                    random_state=int(random_state)
                )

                st.session_state["ml_classification_results"] = results
                st.session_state["ml_classification_config"] = (
                    target, model_name, test_size, int(random_state)
                )

            st.success("Classification model trained successfully.")

        except Exception as error:
            st.error(f"Classification training failed: {error}")

    results = st.session_state.get("ml_classification_results")

    if results is None:
        return

    config = (target, model_name, test_size, int(random_state))

    if config != st.session_state.get("ml_classification_config"):
        st.info("Configuration changed. Train the model again.")
        return

    metrics = results["metrics"]

    st.divider()
    st.markdown("### Model Evaluation")

    cols = st.columns(4)

    for col, name in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ):
        col.metric(name, f"{metrics[name]:.4f}")

    if metrics.get("ROC-AUC") is not None:
        st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

    st.markdown("### Confusion Matrix")

    confusion = pd.DataFrame(
        results["confusion_matrix"],
        index=results["class_names"],
        columns=results["class_names"]
    )

    st.dataframe(confusion, use_container_width=True)
    st.bar_chart(confusion, use_container_width=True)

    st.markdown("### Classification Report")

    report = pd.DataFrame(
        results["classification_report"]
    ).transpose()

    st.dataframe(report, use_container_width=True)

    st.markdown("### Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual": results["actual"],
        "Predicted": results["predicted"]
    })

    st.dataframe(comparison, use_container_width=True)

    importance = get_classification_feature_importance(
        results["model"]
    )

    if importance is not None and not importance.empty:
        st.markdown("### Feature Importance")
        st.bar_chart(
            importance.head(20).set_index("Feature")["Importance"],
            use_container_width=True
        )

    _show_model_download(
        export_classification_model(results["model"]),
        "datalens_classification_model.joblib",
        "ml_download_classification"
    )


# =========================================================
# CLUSTERING
# =========================================================

def render_clustering():

    st.subheader("Clustering Analysis")
    st.write("Discover groups and patterns without a target variable.")

    df = _get_dataset()
    if df is None:
        return

    model_name = st.selectbox(
        "Clustering Algorithm",
        list(get_clustering_models().keys()),
        key="ml_clustering_algorithm"
    )

    excluded = st.multiselect(
        "Exclude Columns",
        df.columns.tolist(),
        key="ml_clustering_excluded"
    )

    n_clusters = 3
    eps = 0.5
    min_samples = 5

    cluster_algorithms = [
        "K-Means",
        "Mini-Batch K-Means",
        "Agglomerative Clustering",
        "Birch",
        "Gaussian Mixture Model",
        "Spectral Clustering"
    ]

    if model_name in cluster_algorithms:
        n_clusters = st.slider(
            "Number of Clusters",
            2, 10, 3,
            key="ml_clustering_count"
        )

    if model_name == "DBSCAN":
        eps = st.number_input(
            "Epsilon",
            0.01, 10.0, 0.5, 0.05,
            key="ml_clustering_eps"
        )

        min_samples = st.number_input(
            "Minimum Samples",
            2, 100, 5,
            key="ml_clustering_min_samples"
        )

    if model_name == "OPTICS":
        min_samples = st.number_input(
            "Minimum Samples",
            2, 100, 5,
            key="ml_optics_min_samples"
        )

    random_state = st.number_input(
        "Random State",
        0, 10000, 42,
        key="ml_clustering_random_state"
    )

    if st.button(
        "Run Clustering",
        type="primary",
        use_container_width=True,
        key="ml_run_clustering"
    ):
        try:
            with st.spinner("Running clustering..."):

                results = train_clustering_model(
                    df=df,
                    model_name=model_name,
                    excluded_columns=excluded,
                    n_clusters=n_clusters,
                    eps=eps,
                    min_samples=int(min_samples),
                    random_state=int(random_state)
                )

                st.session_state["ml_clustering_results"] = results
                st.session_state["ml_clustering_config"] = (
                    model_name,
                    excluded,
                    n_clusters,
                    eps,
                    int(min_samples),
                    int(random_state)
                )

            st.success("Clustering completed successfully.")

        except Exception as error:
            st.error(f"Clustering failed: {error}")

    results = st.session_state.get("ml_clustering_results")

    if results is None:
        return

    config = (
        model_name,
        excluded,
        n_clusters,
        eps,
        int(min_samples),
        int(random_state)
    )

    if config != st.session_state.get("ml_clustering_config"):
        st.info("Configuration changed. Run clustering again.")
        return

    st.divider()
    st.markdown("### Clustering Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Clusters Found", results["number_of_clusters"])
    c2.metric("Observations", results["training_rows"])
    c3.metric("Noise Points", results["noise_count"])

    st.markdown("### Evaluation Metrics")

    metrics = results["metrics"]

    c1, c2, c3 = st.columns(3)

    for col, name in zip(
        [c1, c2, c3],
        [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index"
        ]
    ):
        value = metrics[name]
        col.metric(
            name,
            f"{value:.4f}" if value is not None else "N/A"
        )

    st.markdown("### Cluster Visualization")

    st.scatter_chart(
        results["cluster_data"],
        x="Component 1",
        y="Component 2",
        color="Cluster",
        use_container_width=True
    )

    st.markdown("### Cluster Distribution")

    st.bar_chart(
        results["distribution"].set_index("Cluster")["Observations"],
        use_container_width=True
    )

    with st.expander("Cluster Summary"):
        st.dataframe(
            get_cluster_summary(results),
            use_container_width=True
        )

    st.markdown("### Clustered Dataset")

    st.dataframe(
        results["clustered_dataset"].head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Clustered Dataset",
        data=export_clustered_dataset(results),
        file_name="datalens_clustered_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_download_clustered_csv"
    )

    _show_model_download(
        export_clustering_model(results),
        "datalens_clustering_model.joblib",
        "ml_download_clustering_model"
    )


# =========================================================
# DIMENSIONALITY REDUCTION
# =========================================================

def render_dimensionality_reduction():

    st.subheader("Dimensionality Reduction")
    st.write(
        "Transform high-dimensional datasets into fewer components."
    )

    df = _get_dataset()
    if df is None:
        return

    algorithms = get_dimensionality_algorithms()

    algorithm = st.selectbox(
        "Algorithm",
        list(algorithms.keys()),
        key="ml_dim_algorithm"
    )

    st.info(algorithms[algorithm]["description"])

    excluded = st.multiselect(
        "Exclude Columns",
        df.columns.tolist(),
        key="ml_dim_excluded"
    )

    components = st.radio(
        "Number of Components",
        [2, 3],
        horizontal=True,
        key="ml_dim_components"
    )

    kernel = "rbf"
    gamma = None
    perplexity = 30
    learning_rate = 200
    iterations = 1000
    neighbors = 15
    min_dist = 0.1

    if algorithm == "Kernel PCA":

        kernel = st.selectbox(
            "Kernel",
            ["linear", "poly", "rbf", "sigmoid", "cosine"],
            key="ml_dim_kernel"
        )

        gamma_value = st.number_input(
            "Gamma (0 = default)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            key="ml_dim_gamma"
        )

        gamma = gamma_value if gamma_value > 0 else None

    if algorithm == "t-SNE":

        perplexity = st.number_input(
            "Perplexity",
            2, 100, 30,
            key="ml_dim_perplexity"
        )

        learning_rate = st.number_input(
            "Learning Rate",
            10, 1000, 200,
            key="ml_dim_learning_rate"
        )

        iterations = st.number_input(
            "Iterations",
            250, 5000, 1000, 250,
            key="ml_dim_iterations"
        )

    if algorithm == "UMAP":

        neighbors = st.number_input(
            "Number of Neighbors",
            2, 100, 15,
            key="ml_dim_neighbors"
        )

        min_dist = st.slider(
            "Minimum Distance",
            0.0, 1.0, 0.1, 0.05,
            key="ml_dim_min_dist"
        )

    random_state = st.number_input(
        "Random State",
        0, 10000, 42,
        key="ml_dim_random_state"
    )

    if st.button(
        "Run Dimensionality Reduction",
        type="primary",
        use_container_width=True,
        key="ml_run_dimensionality"
    ):
        try:
            with st.spinner(f"Running {algorithm}..."):

                results = run_dimensionality_reduction(
                    df=df,
                    algorithm=algorithm,
                    n_components=components,
                    excluded_columns=excluded,
                    kernel=kernel,
                    gamma=gamma,
                    perplexity=int(perplexity),
                    learning_rate=int(learning_rate),
                    n_iter=int(iterations),
                    random_state=int(random_state),
                    n_neighbors=int(neighbors),
                    min_dist=float(min_dist)
                )

                st.session_state["ml_dimensionality_results"] = results

                st.session_state["ml_dimensionality_config"] = (
                    algorithm,
                    components,
                    excluded,
                    kernel,
                    gamma,
                    int(perplexity),
                    int(learning_rate),
                    int(iterations),
                    int(random_state),
                    int(neighbors),
                    float(min_dist)
                )

            st.success("Dimensionality reduction completed.")

        except Exception as error:
            st.error(f"Dimensionality reduction failed: {error}")

    results = st.session_state.get("ml_dimensionality_results")

    if results is None:
        return

    config = (
        algorithm,
        components,
        excluded,
        kernel,
        gamma,
        int(perplexity),
        int(learning_rate),
        int(iterations),
        int(random_state),
        int(neighbors),
        float(min_dist)
    )

    if config != st.session_state.get("ml_dimensionality_config"):
        st.info("Configuration changed. Run dimensionality reduction again.")
        return

    st.divider()
    st.markdown("### Reduction Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Original Features", results["original_features"])
    c2.metric("Transformed Features", results["transformed_features"])
    c3.metric("Components", results["number_of_components"])
    c4.metric("Observations", results["training_rows"])

    variance = results["total_explained_variance"]

    if variance is not None:
        st.metric("Explained Variance", f"{variance:.2f}%")
        st.progress(min(max(variance / 100, 0.0), 1.0))
    else:
        st.info(
            "Explained variance is not directly available "
            "for this embedding algorithm."
        )

    reduced = results["reduced_data"]

    st.markdown("### Reduced Data Visualization")

    if components == 2:

        st.scatter_chart(
            reduced,
            x="Component_1",
            y="Component_2",
            use_container_width=True
        )

    else:

        try:
            import plotly.express as px

            fig = px.scatter_3d(
                reduced,
                x="Component_1",
                y="Component_2",
                z="Component_3",
                title=f"{algorithm} - 3D Visualization"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as error:
            st.warning(f"3D chart unavailable: {error}")

    component_summary = get_component_summary(results)

    if component_summary is not None:

        st.markdown("### Component Information")

        st.dataframe(
            component_summary,
            use_container_width=True
        )

        st.bar_chart(
            component_summary.set_index("Component")[
                "Explained Variance (%)"
            ],
            use_container_width=True
        )

    else:
        st.info("Component-wise explained variance is unavailable.")

    st.markdown("### Reduced Dataset")

    st.dataframe(
        results["output_dataset"].head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Reduced Dataset",
        data=export_reduced_dataset(results),
        file_name="datalens_reduced_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_download_dimensionality_csv"
    )

    _show_model_download(
        export_dimensionality_model(results),
        "datalens_dimensionality_model.joblib",
        "ml_download_dimensionality_model"
    )


# =========================================================
# ANOMALY DETECTION
# =========================================================

def render_anomaly_detection():

    st.subheader("Anomaly Detection")

    st.write(
        "Identify unusual observations, outliers and potential "
        "data-quality issues using unsupervised algorithms."
    )

    df = _get_dataset()
    if df is None:
        return

    algorithms = get_anomaly_algorithms()

    algorithm = st.selectbox(
        "Select Anomaly Detection Algorithm",
        list(algorithms.keys()),
        key="ml_anomaly_algorithm"
    )

    st.info(algorithms[algorithm]["description"])

    excluded = st.multiselect(
        "Exclude Columns (Optional)",
        df.columns.tolist(),
        help="Exclude IDs, target labels or irrelevant features.",
        key="ml_anomaly_excluded"
    )

    contamination = 0.05
    n_neighbors = 20
    nu = 0.05
    kernel = "rbf"
    gamma = "scale"
    eps = 0.5
    min_samples = 5

    if algorithm in [
        "Isolation Forest",
        "Local Outlier Factor",
        "Elliptic Envelope"
    ]:

        contamination = st.slider(
            "Expected Anomaly Proportion",
            min_value=0.01,
            max_value=0.50,
            value=0.05,
            step=0.01,
            format="%.2f",
            key="ml_anomaly_contamination"
        )

        st.caption(
            "This is an expected proportion, not a guarantee "
            "that the algorithm will identify exactly that percentage."
        )

    if algorithm == "Local Outlier Factor":

        n_neighbors = st.number_input(
            "Number of Neighbors",
            min_value=2,
            max_value=100,
            value=20,
            key="ml_anomaly_neighbors"
        )

    if algorithm == "One-Class SVM":

        nu = st.slider(
            "Nu",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="ml_anomaly_nu"
        )

        kernel = st.selectbox(
            "Kernel",
            ["rbf", "linear", "poly", "sigmoid"],
            key="ml_anomaly_kernel"
        )

        gamma = st.selectbox(
            "Gamma",
            ["scale", "auto"],
            key="ml_anomaly_gamma"
        )

    if algorithm == "DBSCAN":

        col1, col2 = st.columns(2)

        with col1:
            eps = st.number_input(
                "Epsilon",
                min_value=0.01,
                max_value=10.0,
                value=0.5,
                step=0.05,
                key="ml_anomaly_eps"
            )

        with col2:
            min_samples = st.number_input(
                "Minimum Samples",
                min_value=2,
                max_value=100,
                value=5,
                key="ml_anomaly_min_samples"
            )

    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=10000,
        value=42,
        key="ml_anomaly_random_state"
    )

    st.divider()

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(8),
        use_container_width=True
    )

    if st.button(
        "Detect Anomalies",
        type="primary",
        use_container_width=True,
        key="ml_run_anomaly"
    ):

        try:
            with st.spinner(
                f"Running {algorithm} anomaly detection..."
            ):

                results = run_anomaly_detection(
                    df=df,
                    algorithm=algorithm,
                    excluded_columns=excluded,
                    contamination=float(contamination),
                    n_neighbors=int(n_neighbors),
                    nu=float(nu),
                    kernel=kernel,
                    gamma=gamma,
                    eps=float(eps),
                    min_samples=int(min_samples),
                    random_state=int(random_state)
                )

                st.session_state["ml_anomaly_results"] = results

                st.session_state["ml_anomaly_config"] = (
                    algorithm,
                    excluded,
                    float(contamination),
                    int(n_neighbors),
                    float(nu),
                    kernel,
                    gamma,
                    float(eps),
                    int(min_samples),
                    int(random_state)
                )

            st.success("Anomaly detection completed successfully.")

        except Exception as error:
            st.error(f"Anomaly detection failed: {error}")

    results = st.session_state.get("ml_anomaly_results")

    if results is None:
        return

    config = (
        algorithm,
        excluded,
        float(contamination),
        int(n_neighbors),
        float(nu),
        kernel,
        gamma,
        float(eps),
        int(min_samples),
        int(random_state)
    )

    if config != st.session_state.get("ml_anomaly_config"):
        st.info("Configuration changed. Run anomaly detection again.")
        return

    st.divider()

    st.markdown("### Anomaly Detection Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Observations", results["total_count"])
    c2.metric("Normal Observations", results["normal_count"])
    c3.metric("Anomalies", results["anomaly_count"])
    c4.metric(
        "Anomaly Percentage",
        f"{results['anomaly_percentage']:.2f}%"
    )

    st.divider()

    st.markdown("### Normal vs Anomaly Distribution")

    st.bar_chart(
        results["score_summary"].set_index("Category")[
            "Observations"
        ],
        use_container_width=True
    )

    with st.expander("View distribution details"):
        st.dataframe(
            results["score_summary"],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.markdown("### Anomaly Score Distribution")

    score_df = pd.DataFrame({
        "Anomaly Score": results["anomaly_scores"],
        "Status": results["output_dataset"]["Anomaly_Status"].to_numpy()
    })

    st.scatter_chart(
        score_df,
        x="Anomaly Score",
        y="Anomaly Score",
        color="Status",
        use_container_width=True
    )

    st.caption(
        "Scores are algorithm-specific and should not be compared "
        "directly across different algorithms."
    )

    st.divider()

    st.markdown("### Feature-wise Anomaly Analysis")

    feature_summary = results["feature_summary"]

    if feature_summary is not None and not feature_summary.empty:

        st.dataframe(
            feature_summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Mean differences compare the detected anomaly group "
            "with the normal group. They are descriptive, not causal."
        )

    else:
        st.info(
            "Feature-wise numerical analysis is unavailable, "
            "or no anomalies were identified."
        )

    st.divider()

    st.markdown("### Detected Anomalies")

    anomaly_rows = results["output_dataset"].loc[
        results["output_dataset"]["Anomaly_Label"] == -1
    ]

    st.dataframe(
        anomaly_rows.head(100),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Complete Anomaly-labelled Dataset")

    st.dataframe(
        results["output_dataset"].head(100),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Export Results")

    st.download_button(
        "Download Anomaly-labelled Dataset (CSV)",
        data=export_anomaly_dataset(results),
        file_name="datalens_anomaly_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="ml_download_anomaly_dataset"
    )

    st.download_button(
        "Download Anomaly Summary (CSV)",
        data=export_anomaly_summary(results),
        file_name="datalens_anomaly_summary.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_download_anomaly_summary"
    )

    _show_model_download(
        export_anomaly_model(results),
        "datalens_anomaly_model.joblib",
        "ml_download_anomaly_model"
    )


# =========================================================
# ML-07: ASSOCIATION RULE MINING
# =========================================================

def render_association_mining():

    st.subheader("Association Rule Mining")

    st.write(
        "Discover frequently occurring item combinations and "
        "relationships between items in transactional datasets."
    )

    df = _get_dataset()

    if df is None:
        return

    # -----------------------------------------------------
    # INPUT CONFIGURATION
    # -----------------------------------------------------

    input_format = st.radio(
        "Dataset Format",
        ["Transaction", "One-hot"],
        horizontal=True,
        help=(
            "Transaction: a column contains comma-separated items. "
            "One-hot: each item has its own boolean or 0/1 column."
        ),
        key="ml_association_format"
    )

    item_column = None
    transaction_column = None
    separator = ","
    excluded_columns = []

    if input_format == "Transaction":

        col1, col2 = st.columns(2)

        with col1:
            item_column = st.selectbox(
                "Items Column",
                df.columns.tolist(),
                key="ml_association_item_column"
            )

        with col2:
            transaction_options = [
                "No transaction ID"
            ] + df.columns.tolist()

            transaction_choice = st.selectbox(
                "Transaction ID Column (Optional)",
                transaction_options,
                key="ml_association_transaction_column"
            )

            if transaction_choice != "No transaction ID":
                transaction_column = transaction_choice

        separator = st.text_input(
            "Item Separator",
            value=",",
            help="For example, use a comma for Bread, Milk, Butter.",
            key="ml_association_separator"
        )

        st.caption(
            "If each row contains a complete transaction, leave the "
            "transaction ID unset. If each row contains one item, "
            "select the transaction ID column."
        )

    else:

        excluded_columns = st.multiselect(
            "Exclude Non-item Columns",
            df.columns.tolist(),
            help="Exclude transaction IDs, dates and other metadata.",
            key="ml_association_excluded_columns"
        )

        st.caption(
            "Remaining columns must contain boolean or 0/1 values."
        )

    # -----------------------------------------------------
    # ALGORITHM AND PARAMETERS
    # -----------------------------------------------------

    st.markdown("### Mining Configuration")

    algorithms = get_association_algorithms()

    col1, col2 = st.columns(2)

    with col1:
        algorithm = st.selectbox(
            "Mining Algorithm",
            list(algorithms.keys()),
            key="ml_association_algorithm"
        )

    with col2:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="ml_association_support"
        )

    st.info(algorithms[algorithm]["description"])

    col1, col2, col3 = st.columns(3)

    with col1:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="ml_association_confidence"
        )

    with col2:
        min_lift = st.number_input(
            "Minimum Lift",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="ml_association_lift"
        )

    with col3:
        max_len_choice = st.selectbox(
            "Maximum Itemset Length",
            ["No Limit", 2, 3, 4, 5, 6, 7, 8],
            key="ml_association_max_len"
        )

    max_len = (
        None
        if max_len_choice == "No Limit"
        else int(max_len_choice)
    )

    st.caption(
        "Support measures how often an itemset occurs. Confidence "
        "measures how often the consequent occurs when the antecedent "
        "occurs. Lift compares this relationship with independent "
        "co-occurrence."
    )

    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    with st.expander("Preview Input Dataset"):
        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    # -----------------------------------------------------
    # RUN MINING
    # -----------------------------------------------------

    if st.button(
        "Run Association Rule Mining",
        type="primary",
        use_container_width=True,
        key="ml_run_association"
    ):

        try:

            with st.spinner(
                f"Running {algorithm} association rule mining..."
            ):

                results = run_association_mining(
                    data=df,
                    algorithm=algorithm,
                    input_format=input_format,
                    item_column=item_column,
                    transaction_column=transaction_column,
                    separator=separator,
                    excluded_columns=excluded_columns,
                    min_support=float(min_support),
                    min_confidence=float(min_confidence),
                    min_lift=float(min_lift),
                    max_len=max_len
                )

                st.session_state["ml_association_results"] = results

                st.session_state["ml_association_config"] = (
                    input_format,
                    item_column,
                    transaction_column,
                    separator,
                    tuple(excluded_columns),
                    algorithm,
                    float(min_support),
                    float(min_confidence),
                    float(min_lift),
                    max_len
                )

            st.success(
                "Association Rule Mining completed successfully."
            )

        except Exception as error:
            st.error(f"Association Rule Mining failed: {error}")

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    results = st.session_state.get("ml_association_results")

    if results is None:
        return

    current_config = (
        input_format,
        item_column,
        transaction_column,
        separator,
        tuple(excluded_columns),
        algorithm,
        float(min_support),
        float(min_confidence),
        float(min_lift),
        max_len
    )

    if current_config != st.session_state.get(
        "ml_association_config"
    ):
        st.info(
            "Configuration changed. Run Association Rule Mining again "
            "to update the results."
        )
        return

    summary = results["summary"]

    st.divider()

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    st.markdown("### Mining Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions",
        f"{summary['transaction_count']:,}"
    )

    c2.metric(
        "Unique Items",
        f"{summary['item_count']:,}"
    )

    c3.metric(
        "Frequent Itemsets",
        f"{summary['frequent_itemset_count']:,}"
    )

    c4.metric(
        "Association Rules",
        f"{summary['association_rule_count']:,}"
    )

    st.caption(
        f"Algorithm: {summary['algorithm']} | "
        f"Input format: {summary['input_format']}"
    )

    st.divider()

    # -----------------------------------------------------
    # FREQUENT ITEMSETS
    # -----------------------------------------------------

    st.markdown("### Frequent Itemsets")

    st.write(
        "Item combinations that occur frequently across transactions."
    )

    itemsets = results["itemsets_output"]

    if itemsets.empty:

        st.info("No frequent itemsets are available.")

    else:

        st.dataframe(
            itemsets,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Most Frequent Itemsets")

        chart_data = itemsets.head(15).copy()

        st.bar_chart(
            chart_data.set_index("Itemset")["Support"],
            use_container_width=True
        )

    st.download_button(
        "Download Frequent Itemsets (CSV)",
        data=export_frequent_itemsets(results),
        file_name="datalens_frequent_itemsets.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_download_association_itemsets"
    )

    st.divider()

    # -----------------------------------------------------
    # ASSOCIATION RULES
    # -----------------------------------------------------

    st.markdown("### Association Rules")

    rules = results["rules_output"]

    if rules.empty:

        st.warning(
            "No association rules satisfy the current confidence "
            "and lift thresholds. Try reducing these thresholds."
        )

    else:

        st.write(
            "Rules show which items are associated with other items."
        )

        st.markdown("#### Filter Rules")

        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            support_filter = st.slider(
                "Rule Support",
                0.0,
                1.0,
                float(min_support),
                0.01,
                key="ml_association_filter_support"
            )

        with fc2:
            confidence_filter = st.slider(
                "Rule Confidence",
                0.0,
                1.0,
                float(min_confidence),
                0.05,
                key="ml_association_filter_confidence"
            )

        with fc3:
            lift_filter = st.number_input(
                "Rule Lift",
                min_value=0.0,
                max_value=100.0,
                value=float(min_lift),
                step=0.1,
                key="ml_association_filter_lift"
            )

        filtered_rules = filter_association_rules(
            rules,
            minimum_support=float(support_filter),
            minimum_confidence=float(confidence_filter),
            minimum_lift=float(lift_filter)
        )

        st.caption(
            f"Displaying {len(filtered_rules):,} of "
            f"{len(rules):,} generated rules."
        )

        st.dataframe(
            filtered_rules,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Top Association Rules")

        if not filtered_rules.empty:

            top_rules = filtered_rules.head(15).copy()

            st.bar_chart(
                top_rules.set_index("Rule")["Lift"],
                use_container_width=True
            )

            st.markdown("#### Rule Metric Summary")

            metric_summary = filtered_rules[
                [
                    "Support",
                    "Confidence",
                    "Lift",
                    "Leverage",
                    "Conviction"
                ]
            ].describe().transpose()

            st.dataframe(
                metric_summary,
                use_container_width=True
            )

        else:

            st.info(
                "No rules match the additional filters. "
                "Try lowering the filter thresholds."
            )

    st.download_button(
        "Download Association Rules (CSV)",
        data=export_association_rules(results),
        file_name="datalens_association_rules.csv",
        mime="text/csv",
        use_container_width=True,
        key="ml_download_association_rules"
    )

    st.divider()

    # -----------------------------------------------------
    # EXPORT MODEL / RESULTS
    # -----------------------------------------------------

    st.markdown("### Export Mining Results")

    st.write(
        "Export the mining configuration, frequent itemsets, "
        "rules and transaction encoder for later reference."
    )

    _show_model_download(
        export_association_model(results),
        "datalens_association_mining.joblib",
        "ml_download_association_model"
    )


# =========================================================
# MAIN MACHINE LEARNING STUDIO
# =========================================================

def render_machine_learning():

    st.title("🤖 Machine Learning Studio")

    st.write(
        "Train, evaluate and explore machine learning models "
        "using your DataLens dataset."
    )

    df = _get_dataset()

    if df is None:
        return

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.subheader("Dataset Overview")

    info = inspect_dataset(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Features", f"{info['columns']:,}")
    c3.metric("Missing Values", f"{info['missing_values']:,}")
    c4.metric(
        "Algorithms in Catalogue",
        get_algorithm_count()
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET READINESS
    # -----------------------------------------------------

    st.subheader("Dataset Readiness")

    readiness = get_dataset_readiness(df)

    if readiness["ready"]:
        st.success("No basic data-readiness issues detected.")
    else:
        for warning in readiness["warnings"]:
            st.warning(warning)

    st.divider()

    # -----------------------------------------------------
    # TASK SETUP
    # -----------------------------------------------------

    st.subheader("Machine Learning Task Setup")

    target = st.selectbox(
        "Select Target Variable",
        df.columns.tolist(),
        key="ml_foundation_target"
    )

    st.info(
        f"Suggested ML task: {suggest_ml_task(df, target)}"
    )

    st.caption(
        "Task suggestions are preliminary. Clustering, "
        "dimensionality reduction and anomaly detection "
        "do not require a target variable."
    )

    st.divider()

    # -----------------------------------------------------
    # ALGORITHM LIBRARY
    # -----------------------------------------------------

    st.subheader("Algorithm Library")

    family = st.selectbox(
        "Select Algorithm Family",
        get_algorithm_families(),
        key="ml_algorithm_family"
    )

    algorithms = get_algorithms(family)

    names = list(algorithms.keys())

    st.caption(f"{len(names)} algorithms in this family")

    for index, name in enumerate(names, start=1):
        st.markdown(f"**{index}.** {name}")

    st.divider()

    # -----------------------------------------------------
    # ML-02: REGRESSION
    # -----------------------------------------------------

    with st.expander("Regression Analysis", expanded=True):
        render_regression()

    # -----------------------------------------------------
    # ML-03: CLASSIFICATION
    # -----------------------------------------------------

    with st.expander("Classification Analysis", expanded=False):
        render_classification()

    # -----------------------------------------------------
    # ML-04: CLUSTERING
    # -----------------------------------------------------

    with st.expander("Clustering Analysis", expanded=False):
        render_clustering()

    # -----------------------------------------------------
    # ML-05: DIMENSIONALITY REDUCTION
    # -----------------------------------------------------

    with st.expander(
        "Dimensionality Reduction",
        expanded=False
    ):
        render_dimensionality_reduction()

    # -----------------------------------------------------
    # ML-06: ANOMALY DETECTION
    # -----------------------------------------------------

    with st.expander(
        "Anomaly Detection",
        expanded=False
    ):
        render_anomaly_detection()

    # -----------------------------------------------------
    # ML-07: ASSOCIATION RULE MINING
    # -----------------------------------------------------

    with st.expander(
        "Association Rule Mining",
        expanded=False
    ):
        render_association_mining()

    st.divider()

    # -----------------------------------------------------
    # UPCOMING MODULES
    # -----------------------------------------------------

    st.subheader("Upcoming ML Modules")

    st.write(
        """
        - Time-Series Forecasting
        - Deep Learning
        - Model Evaluation and Comparison
        - AutoML
        - Hyperparameter Tuning
        - Explainable AI
        - Model Management
        """
    )
