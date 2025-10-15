import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_and_preprocess(self):
        self.df = pd.read_csv(self.data_path)
        # Replace zeroes with NaN in specific columns
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.df[cols_with_zero] = self.df[cols_with_zero].replace(0, np.nan)
        self.df.fillna(self.df.median(), inplace=True)

        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    def train_models(self):
        # Logistic Regression
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = log_reg

        # K-Nearest Neighbors with GridSearch
        knn = KNeighborsClassifier()
        knn_params = {'n_neighbors': range(3, 11)}
        knn_grid = GridSearchCV(knn, knn_params, cv=5)
        knn_grid.fit(self.X_train, self.y_train)
        self.models['KNN'] = knn_grid.best_estimator_

        # Naïve Bayes
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = nb

    def evaluate_models(self):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None

            self.results[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred),
                'ROC AUC': roc_auc_score(self.y_test, y_prob) if y_prob is not None else None,
                'Confusion Matrix': confusion_matrix(self.y_test, y_pred),
                'Classification Report': classification_report(self.y_test, y_pred, output_dict=True)
            }

    def compare_models(self):
        print("\nModel Comparison:")
        summary = pd.DataFrame(self.results).T[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']]
        print(summary.round(3))

    def plot_confusion_matrices(self):
        for name, metrics in self.results.items():
            cm = metrics['Confusion Matrix']
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()

