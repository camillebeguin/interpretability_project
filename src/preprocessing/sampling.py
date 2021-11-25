from imblearn.over_sampling import SMOTENC
import constants as cst


def smotenc(X_train, y_train):
    """
    Synthetic Minority Over-sampling Technique for Nominal and Continuous
    Args:
        X_train: X_train
        y_train: y_train

    Returns: resampled train data (X and y)
    """
    sampling_strategy = cst.sampling_strategy  # desired ratio of resampling
    if not isinstance(sampling_strategy, str) and not isinstance(
        sampling_strategy, float
    ):
        raise TypeError("Sampling strategy must be float or string.")

    random_state = cst.random_state
    categorical_indices = [
        list(X_train.columns).index(i) for i in cst.categorical_columns
    ]
    sm = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled
