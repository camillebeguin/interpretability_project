import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def ordinal_encoding(X_train, X_test, spec_conf):
    ordinal_encoding_dic = spec_conf["preprocessing"]["ordinal_encoding"]
    ordinal_encoding_columns = list(ordinal_encoding_dic.keys())

    if ordinal_encoding_columns:
        for ordinal_col in ordinal_encoding_columns:
            X_train.loc[:, ordinal_col] = X_train.loc[:, ordinal_col].map(ordinal_encoding_dic[ordinal_col])
            X_test.loc[:, ordinal_col] = X_test.loc[:, ordinal_col].map(ordinal_encoding_dic[ordinal_col])
    
    return X_train, X_test


def get_train_test(df, spec_conf):
    test_size = spec_conf["preprocessing"]["train_test_split"]["test_size"]
    random_state =  spec_conf["preprocessing"]["train_test_split"]["random_state"]
    X, y = split_data(df, spec_conf)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test


def split_data(df, spec_conf):
    target_column_name = spec_conf["target_column_name"]
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]
    return X, y

