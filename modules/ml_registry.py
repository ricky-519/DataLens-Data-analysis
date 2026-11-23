
"""
DataLens Machine Learning Studio
Central registry of machine learning algorithms.
"""

ML_ALGORITHMS = {
    "Regression": {
        "Linear Regression": "sklearn.linear_model.LinearRegression",
        "Ridge Regression": "sklearn.linear_model.Ridge",
        "Lasso Regression": "sklearn.linear_model.Lasso",
        "Elastic Net": "sklearn.linear_model.ElasticNet",
        "Polynomial Regression": "sklearn.preprocessing.PolynomialFeatures",
        "Bayesian Ridge": "sklearn.linear_model.BayesianRidge",
        "Huber Regression": "sklearn.linear_model.HuberRegressor",
        "Decision Tree Regressor": "sklearn.tree.DecisionTreeRegressor",
        "Random Forest Regressor": "sklearn.ensemble.RandomForestRegressor",
        "Extra Trees Regressor": "sklearn.ensemble.ExtraTreesRegressor",
        "Gradient Boosting Regressor": "sklearn.ensemble.GradientBoostingRegressor",
        "AdaBoost Regressor": "sklearn.ensemble.AdaBoostRegressor",
        "Hist Gradient Boosting Regressor": "sklearn.ensemble.HistGradientBoostingRegressor",
        "Support Vector Regressor": "sklearn.svm.SVR",
        "KNN Regressor": "sklearn.neighbors.KNeighborsRegressor",
        "XGBoost Regressor": "xgboost.XGBRegressor",
        "LightGBM Regressor": "lightgbm.LGBMRegressor",
        "CatBoost Regressor": "catboost.CatBoostRegressor"
    },

    "Classification": {
        "Logistic Regression": "sklearn.linear_model.LogisticRegression",
        "KNN Classifier": "sklearn.neighbors.KNeighborsClassifier",
        "Decision Tree Classifier": "sklearn.tree.DecisionTreeClassifier",
        "Random Forest Classifier": "sklearn.ensemble.RandomForestClassifier",
        "Extra Trees Classifier": "sklearn.ensemble.ExtraTreesClassifier",
        "Random Forest": "sklearn.ensemble.RandomForestClassifier",
        "Gradient Boosting Classifier": "sklearn.ensemble.GradientBoostingClassifier",
        "AdaBoost Classifier": "sklearn.ensemble.AdaBoostClassifier",
        "Hist Gradient Boosting Classifier": "sklearn.ensemble.HistGradientBoostingClassifier",
        "Support Vector Classifier": "sklearn.svm.SVC",
        "Gaussian Naive Bayes": "sklearn.naive_bayes.GaussianNB",
        "Multinomial Naive Bayes": "sklearn.naive_bayes.MultinomialNB",
        "Bernoulli Naive Bayes": "sklearn.naive_bayes.BernoulliNB",
        "Linear Discriminant Analysis": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        "Quadratic Discriminant Analysis": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        "XGBoost Classifier": "xgboost.XGBClassifier",
        "LightGBM Classifier": "lightgbm.LGBMClassifier",
        "CatBoost Classifier": "catboost.CatBoostClassifier"
    },

    "Clustering": {
        "K-Means": "sklearn.cluster.KMeans",
        "Mini Batch K-Means": "sklearn.cluster.MiniBatchKMeans",
        "DBSCAN": "sklearn.cluster.DBSCAN",
        "OPTICS": "sklearn.cluster.OPTICS",
        "Agglomerative Clustering": "sklearn.cluster.AgglomerativeClustering",
        "Birch": "sklearn.cluster.Birch",
        "Mean Shift": "sklearn.cluster.MeanShift",
        "Spectral Clustering": "sklearn.cluster.SpectralClustering",
        "Gaussian Mixture Model": "sklearn.mixture.GaussianMixture",
        "Affinity Propagation": "sklearn.cluster.AffinityPropagation"
    },

    "Dimensionality Reduction": {
        "PCA": "sklearn.decomposition.PCA",
        "Incremental PCA": "sklearn.decomposition.IncrementalPCA",
        "Kernel PCA": "sklearn.decomposition.KernelPCA",
        "Truncated SVD": "sklearn.decomposition.TruncatedSVD",
        "Factor Analysis": "sklearn.decomposition.FactorAnalysis",
        "Fast ICA": "sklearn.decomposition.FastICA",
        "NMF": "sklearn.decomposition.NMF",
        "Isomap": "sklearn.manifold.Isomap",
        "Locally Linear Embedding": "sklearn.manifold.LocallyLinearEmbedding",
        "t-SNE": "sklearn.manifold.TSNE",
        "UMAP": "umap.UMAP"
    },

    "Anomaly Detection": {
        "Isolation Forest": "sklearn.ensemble.IsolationForest",
        "Local Outlier Factor": "sklearn.neighbors.LocalOutlierFactor",
        "One Class SVM": "sklearn.svm.OneClassSVM",
        "Elliptic Envelope": "sklearn.covariance.EllipticEnvelope",
        "Robust Covariance": "sklearn.covariance.MinCovDet"
    },

    "Association Rules": {
        "Apriori": "mlxtend.frequent_patterns.apriori",
        "FP-Growth": "mlxtend.frequent_patterns.fpgrowth",
        "Association Rules": "mlxtend.frequent_patterns.association_rules"
    },

    "Time-Series Forecasting": {
        "AR": "statsmodels.tsa.ar_model.AutoReg",
        "ARIMA": "statsmodels.tsa.arima.model.ARIMA",
        "SARIMA": "statsmodels.tsa.statespace.sarimax.SARIMAX",
        "Exponential Smoothing": "statsmodels.tsa.holtwinters.ExponentialSmoothing",
        "Prophet": "prophet.Prophet"
    },

    "Deep Learning": {
        "MLP Neural Network": "sklearn.neural_network.MLPClassifier",
        "MLP Regressor": "sklearn.neural_network.MLPRegressor",
        "TensorFlow ANN": "tensorflow.keras",
        "PyTorch Neural Network": "torch.nn",
        "CNN": "tensorflow.keras.layers.Conv2D",
        "RNN": "tensorflow.keras.layers.SimpleRNN",
        "LSTM": "tensorflow.keras.layers.LSTM",
        "GRU": "tensorflow.keras.layers.GRU",
        "Autoencoder": "tensorflow.keras.Model"
    }
}


def get_algorithm_families():
    return list(ML_ALGORITHMS.keys())


def get_algorithms(family):
    return ML_ALGORITHMS.get(family, {})


def get_algorithm_count():
    return sum(len(models) for models in ML_ALGORITHMS.values())


def get_all_algorithms():
    return ML_ALGORITHMS
