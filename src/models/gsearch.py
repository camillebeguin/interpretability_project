#####  Imports  #####
import logging
from typing import Tuple
import pandas as pd

from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBClassifier

import src.constants as cst

logger = logging.getLogger(__name__)


def randomized_search_from_xgb_classifier(
    pipeline,
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[dict, float]:
    """Function to run a grid search for a XGBClassifier

    Args:
        X_train (pd.DataFrame): X_train
        y_train (pd.Series): y_train

    Returns:
        dict: best parameters
        float: best score
    """
    #pip = XGBClassifier(eval_metric="logloss", use_label_encoder=False) 
    best_estimator, best_score = randomized_search_from_estimator_and_params(X_train, y_train, pipeline, cst.xgb_param_dist)
    return best_estimator, best_score


def randomized_search_from_estimator_and_params(
    X_train: pd.DataFrame, y_train: pd.Series, estimator, params_dist: dict
) -> Tuple:
    """
    Main function to run a grid search
    Args:
        X_train (pd.DataFrame): X_train
        y_train (pd.Series): y_train
        estimator (model): prediction model to search parameters for
        params_dist (dict): parameters grid to test in a randomized grid search
    Returns:
        clf: best estimator
        float: best score
    """

    gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=y_train)

    gsearch = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params_dist,
        cv=gkf,
        n_iter=20,
        scoring="roc_auc",
        error_score=0,
        verbose=-1,
        n_jobs=-1,
    )

    best_model = gsearch.fit(X=X_train, y=y_train)
    return best_model.best_estimator_, best_model.best_score_
