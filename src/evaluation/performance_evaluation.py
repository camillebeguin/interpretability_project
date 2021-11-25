#####  Imports  #####
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import src.constants as cst

logger = logging.getLogger(__name__)


def custom_cross_evaluate(
    estimator, X: pd.DataFrame, y: pd.Series, cv=3, metric=accuracy_score
):
    kfolds = StratifiedKFold(n_splits=cv).split(X, y)
    scores = []

    for train_idx, test_idx in kfolds:
        X_train, y_train = X.loc[train_idx, :], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx, :], y.loc[test_idx]

        X_train_encoded, X_test_encoded = X_train, X_test  # processing function
        estimator.fit(X_train_encoded, y_train)
        y_pred = estimator.predict(X_test_encoded)
        scores.append(metric(y_test, y_pred))

    return scores


def compute_performance_metrics(y_true, y_pred):
    most_common_class_count = np.sum(np.array(y_true) == cst.most_common_class)
    benchmark_score = most_common_class_count / len(y_true)

    metrics = pd.DataFrame(
        {
            "benchmark": benchmark_score,
            "accuracy": [accuracy_score(y_true, y_pred)],
            "precision": [precision_score(y_true, y_pred)],
            "recall": [recall_score(y_true, y_pred)],
            "F1 score": [f1_score(y_true, y_pred)],
            "ROC AUC score": [roc_auc_score(y_true, y_pred)],
        }
    )
    return metrics
