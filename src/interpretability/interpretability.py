from PyALE import ale
from pdpbox import pdp
import pandas as pd


def plot_ale_effect(estimator, X_train: pd.DataFrame, feature: str, grid_size=20):
    """Returns ale plot

    Args:
        estimator ([type]): model
        X_train (pd.DataFrame): dataframe of X training set
        feature (str): feature for which you want to plot ALE
        grid_size (int, optional): [description]. Defaults to 20.

    Returns:
        plot: returns plot
    """
    ale_eff = ale(
        X=X_train,
        model=estimator,
        feature=[feature],
        grid_size=grid_size,
        include_CI=True,
        C=0.9,
    )

def plot_univariate_pdp(estimator, X_train: pd.DataFrame, feature: str, feature_name):
    """
    Plots the PDP isolation plot
    Args:
        estimator ([type]): model
        X_train (pd.DataFrame): dataframe of X training set
        feature (str): feature for which you want to plot ALE
        feature_name (str): feature name to appear in the legend
    """
    pdp_feature = pdp.pdp_isolate(
        model=estimator, 
        dataset=X_train, 
        model_features=X_train.columns, 
        feature=feature, 
        num_grid_points=20
    )

    fig, _ = pdp.pdp_plot(pdp_isolate_out=pdp_feature, feature_name=feature_name)
    fig.show()