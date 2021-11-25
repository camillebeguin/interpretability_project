from PyALE import ale
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
