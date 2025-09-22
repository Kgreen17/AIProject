import pandas as pd
import numpy as np

def data_cleaning_and_processing():
    # // upload the dataset csv file
    df = pd.read_csv('covid_mortality.csv')

    # Convert Categorical Variables: Encode Gender ("Male", "Female" → 1, 0)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Handle Missing Values: Fill missing Age values with the mean age
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    print("Missing values per column:")
    # print(df.isnull().sum())

    # remove the missing values
    df.dropna(inplace=True)
    print(df.isnull().sum())

    # detect outliers using IQR method
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
    print(f"Data shape after removing outliers: {df.shape}")

    # detect outliers using visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.boxplot(x=df['Age'])
    plt.title('Boxplot of Age')
    plt.show()

    # plot feature distributions using histograms
    df.hist(figsize=(12, 8))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

    # Visualize the correlation heatmap to see relationships between features.
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Check the class balance of the Death variable.
    print("Class distribution of Death variable:")
    print(df['Death'].value_counts(normalize=True))

    # Split the data into training & testing sets (80/20).
    from sklearn.model_selection import train_test_split
    X = df.drop('Death', axis=1)
    y = df['Death']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # Standardize the features using StandardScaler.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed.")

    #Train a LogisticRegression model with the training data.
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    print("Model training completed.")

    # Make predictions on the test set.
    y_pred = model.predict(X_test_scaled)
    print("Predictions on test set completed.")

    #Assess model performance using key metrics:
        #Confusion Matrix (Visualize with seaborn.heatmap()).
        #Accuracy Score → Overall correctness.
        #Recall Score → How well the model detects actual deaths! (Key metric! 🔥).
        #F1-Score → Balance between precision & recall.
        #ROC-AUC Score → How well the model separates survival vs. death cases.
        #Feature Importance (Logistic Regression Coefficients) → Which biomarkers
        #have the highest impact on mortality?

    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    #  Save & Deploy Your Model
    # 🔹 Save your trained model using pickle or joblib.
    # 🔹 Load the saved model & make predictions on new patient data.
    # 🔹 Test your model: If a patient has a high probability of death, doctors can prioritize
    # treatment!
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    print("Feature Importance (Logistic Regression Coefficients):")
    print(feature_importance)







if __name__ == '__main__':
    data_cleaning_and_processing()