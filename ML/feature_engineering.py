import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_feature_engineering_dataset(file_path, output_path):
    #Load & Explore the Dataset
    df = pd.read_csv(file_path)
    print("Initial Data Snapshot:")
    print(df.head())
    print("Missing Values:")
    print(df.isnull().sum())
    print("Summary Statistics:")
    print(df.describe())

    #Handling Missing Values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    #Detecting and Removing Outliers (IQR Method)
    numerical_cols = ['Age', 'Salary', 'Bonus']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    #Visualize outliers
    for col in numerical_cols:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

    #Feature Scaling
    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()
    for col in numerical_cols:
        df[f'{col}_std'] = scaler_std.fit_transform(df[[col]])
        df[f'{col}_norm'] = scaler_minmax.fit_transform(df[[col]])

    #Encoding Categorical Variables
    if 'Department' in df.columns:
        le = LabelEncoder()
        df['Department_encoded'] = le.fit_transform(df['Department'])

    if 'City' in df.columns:
        df = pd.get_dummies(df, columns=['City'], drop_first=False)

    if 'Education' in df.columns:
        df = pd.get_dummies(df, columns=['Education'], drop_first=True)

    #Final Data Preparation & Saving
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")


