import pandas as pd
from pycaret.classification import *
import os
#import numpy as np
#import matplotlib



def data_loading():
    current_path_str = os.getcwd()
    current_path_list = current_path_str.split("/")
    dataset_path_list = current_path_list[:-1]
    dataset_path_list.append("Dataset")
    dataset_path_str = "/".join(dataset_path_list)
    path = dataset_path_str + "/diabetes.csv"
    df = pd.read_csv(path)
    return df


def model_selection(df):
    experiment = setup(df, target="Outcome")
    best_model = compare_models()
    return best_model


def model_inferrence(model, df):
    return predict_model(model, df)

def model_saving(model, model_name_str):
    save_model(model, model_name = model_name_str)


if __name__ == "__main__":
    df = data_loading()
    best_model = model_selection(df)
    model_inferrence(best_model, df.tail())
    model_saving(best_model, "test")