import json
import pandas as pd
import os

def load_input_data(input_file_name):
    print(os.path.join("data", input_file_name))
    df = pd.read_excel(os.path.join(".\data", input_file_name), engine="openpyxl")
    return df

def load_config_file(config_file_name):
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    return config
