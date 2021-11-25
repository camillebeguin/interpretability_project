from imblearn.over_sampling import SMOTENC
import src.constants as cst

def smotenc(X_train, y_train):
    """
    Synthetic Minority Over-sampling Technique for Nominal and Continuous
    Args:
        X_train: X_train
        y_train: y_train

    Returns: resampled train data (X and y)
    """

    sampling_ratio = cst.classification_model_parameters["sampling"]["sampling_ratio"]
    if not isinstance(sampling_ratio, str) and not isinstance(sampling_ratio, float):
        raise TypeError("Sampling strategy must be float or string.")

    categorical_indices = [list(X_train.columns).index(i) for i in cst.classification_model_parameters["categorical_columns"]]
    sm = SMOTENC(categorical_features=categorical_indices, sampling_strategy=sampling_ratio, random_state=cst.random_state)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled
