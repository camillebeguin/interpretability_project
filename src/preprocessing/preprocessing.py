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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test


def split_data(df, spec_conf):
    X = df.drop(columns=[spec_conf["target_column_name"]]).drop(columns=spec_conf["drop_columns"])
    y = df[spec_conf["target_column_name"]]
    return X, y


def one_hot_encoding(df_train, df_test, spec_conf):
    one_hot_encoder = OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")

    one_hot_columns = spec_conf["preprocessing"]["one_hot_encoding"]["columns"]
    encoded_columns_train = one_hot_encoder.fit_transform(df_train[one_hot_columns])
    df_encoded_columns_train = pd.DataFrame(
        encoded_columns_train, columns=one_hot_encoder.get_feature_names(one_hot_columns)
    )
    df_train_encoded = pd.concat([df_train, df_encoded_columns_train], axis=1).drop(
        columns=one_hot_columns
    )

    encoded_columns_test = one_hot_encoder.transform(df_test[one_hot_columns])
    df_encoded_columns_test = pd.DataFrame(
        encoded_columns_test, columns=one_hot_encoder.get_feature_names(one_hot_columns)
    )
    df_test_encoded = pd.concat([df_test, df_encoded_columns_test], axis=1).drop(
        columns=one_hot_columns
    )
    return df_train_encoded, df_test_encoded


def standard_scaling(df_train, df_test, standard_scale_columns):
    standard_scaler = StandardScaler()
    df_train[standard_scale_columns] = standard_scaler.fit_transform(df_train[standard_scale_columns])
    df_test[standard_scale_columns] = standard_scaler.transform(df_test[standard_scale_columns])
    return df_train, df_test


def min_max_scaling(df_train, df_test, spec_conf):
    min_max_columns = spec_conf["preprocessing"]["min_max_scaling"]["columns"]
    min_max_scaler = MinMaxScaler()
    df_train[min_max_columns] = min_max_scaler.fit_transform(df_train[min_max_columns])
    df_test[min_max_columns] = min_max_scaler.transform(df_test[min_max_columns])
    return df_train, df_test
