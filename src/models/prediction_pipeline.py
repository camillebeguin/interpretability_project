
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline


def build_prediction_pipeline(estimator, spec_conf):
    """
    Builds a pipeline to combine the column encoder and the estimator
    Args:
        estimator: prediction model
    """
    standard_scaling_columns = spec_conf["preprocessing"]["standard_scaling"]["columns"]
    min_max_columns = spec_conf["preprocessing"]["min_max_scaling"]["columns"]
    one_hot_columns = spec_conf["preprocessing"]["one_hot_encoding"]["columns"]
    
    encoder = create_column_encoder(
        one_hot_columns, min_max_columns, standard_scaling_columns
    )

    pipeline = Pipeline(
        (
            [
                ("encoder", encoder),
                ("estimator", estimator),
            ]
        )
    )
    return pipeline


def create_column_encoder(one_hot_columns, min_max_columns, standard_scaling_columns):
    """
    Encodes categorical columns and scales numerical columns
    Args:
        categorical_columns (list): list of categorical columns
    """
    encoder = make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore'), one_hot_columns),
        (MinMaxScaler(), min_max_columns),
        (StandardScaler(), standard_scaling_columns),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return encoder


def standard_scaling(df_train, df_test, standard_scale_columns):
    standard_scaler = StandardScaler()
    df_train[standard_scale_columns] = standard_scaler.fit_transform(
        df_train[standard_scale_columns]
    )
    df_test[standard_scale_columns] = standard_scaler.transform(
        df_test[standard_scale_columns]
    )
    return df_train, df_test


def min_max_scaling(df_train, df_test, min_max_columns):
    min_max_scaler = MinMaxScaler()
    df_train[min_max_columns] = min_max_scaler.fit_transform(df_train[min_max_columns])
    df_test[min_max_columns] = min_max_scaler.transform(df_test[min_max_columns])
    return df_train, df_test
