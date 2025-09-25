
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (

    mean_absolute_error, mean_squared_error, r2_score,

    confusion_matrix, classification_report, roc_auc_score, roc_curve

)


def knn_linear_regression_classification():
    # Load Dataset
    df = pd.read_csv('stock_data.csv')  # Replace with your actual file

    # Data Preprocessing

    df.fillna(df.mean(), inplace=True)  # Handle missing values

    # Create Classification Target

    df['Price_Increase'] = (df['Close'] > df['Open']).astype(int)

    # Exploratory Data Analysis

    df.hist(bins=30, figsize=(15, 10))

    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(10, 8))

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    plt.title("Feature Correlation Heatmap")

    plt.show()

    #Feature Scaling

    features = ['Open', 'High', 'Low', 'Volume', 'Market Sentiment', 'Interest Rate',

                'Inflation', 'GDP Growth', 'Oil Price', 'Gold Price']

    X = df[features]

    y_reg = df['Close']

    y_clf = df['Price_Increase']

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    #Train-Test Split

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

    #KNN Regression + GridSearchCV

    knn_params = {'n_neighbors': list(range(3, 21))}

    knn = KNeighborsRegressor()

    grid_knn = GridSearchCV(knn, knn_params, cv=5, scoring='neg_mean_squared_error')

    grid_knn.fit(X_train_reg, y_train_reg)

    best_knn = grid_knn.best_estimator_

    y_pred_knn = best_knn.predict(X_test_reg)

    #KNN Regression Metrics

    print("KNN Regression Metrics:")

    print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_knn):.2f}")

    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_knn):.2f}")

    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_knn)):.2f}")

    print(f"R²: {r2_score(y_test_reg, y_pred_knn):.2f}")

    plt.scatter(y_test_reg, y_pred_knn, alpha=0.5)

    plt.xlabel("Actual Close Price")

    plt.ylabel("Predicted Close Price")

    plt.title("KNN: Actual vs Predicted")

    plt.show()

    #Linear Regression Models

    models = {

        'Lasso': Lasso(),

        'Ridge': Ridge(),

        'ElasticNet': ElasticNet()

    }

    params = {

        'Lasso': {'alpha': [0.01, 0.1, 1, 10]},

        'Ridge': {'alpha': [0.01, 0.1, 1, 10]},

        'ElasticNet': {'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]}

    }

    for name, model in models.items():

        grid = GridSearchCV(model, params[name], cv=5, scoring='neg_mean_squared_error')

        grid.fit(X_train_reg, y_train_reg)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test_reg)

        print(f"{name} Regression Metrics:")

        print(f"MAE: {mean_absolute_error(y_test_reg, y_pred):.2f}")

        print(f"MSE: {mean_squared_error(y_test_reg, y_pred):.2f}")

        print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred)):.2f}")

        print(f"R²: {r2_score(y_test_reg, y_pred):.2f}")

        residuals = y_test_reg - y_pred

        sns.histplot(residuals, kde=True)

        plt.title(f"{name} Residuals Distribution")

        plt.xlabel("Residuals")

        plt.show()

    #Logistic Regression for Classification

    log_reg = LogisticRegression(max_iter=1000)

    log_reg.fit(X_train_clf, y_train_clf)

    y_pred_clf = log_reg.predict(X_test_clf)

    y_proba_clf = log_reg.predict_proba(X_test_clf)[:, 1]

    print("Logistic Regression Classification Report:")

    print(classification_report(y_test_clf, y_pred_clf))

    cm = confusion_matrix(y_test_clf, y_pred_clf)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()

    fpr, tpr, _ = roc_curve(y_test_clf, y_proba_clf)

    roc_auc = roc_auc_score(y_test_clf, y_proba_clf)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], 'k--')

    plt.title("ROC Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.show()

if __name__ == '__main__':
    knn_linear_regression_classification()