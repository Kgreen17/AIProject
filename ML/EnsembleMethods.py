
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import boxplot
from scipy.misc import dataset_methods
from sklearn.feature_selection import mutual_info_classif


# 1. load the  dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# 2. data overview and statistics
def data_overview(df):
    print("data set info")
    print(df.info())
    print("summary statistics")
    print(df.describe())

    # drop id column as it is not useful for analysis
    df = df.drop(columns=['ID'])

    # check for missing values
    print("missing values in each column")
    print(df.isnull().sum())

    # detect outliners using boxplot
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df.drop(columns=["default payment next month"]))
    plt.title("Boxplot of Feature detectiong outliers")
    plt.xticks(rotation=90)
    plt.show()
if __name__ == '__main__':
    df = load_data('credit card clients.csv')
    data_overview(df)