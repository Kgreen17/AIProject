import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

class BMIPredictor:
    def __init__(self, filepath):
        """
        Initialize the class with the dataset path.
        """
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """
        Load dataset from CSV and display basic info.
        """
        self.df = pd.read_csv(self.filepath)
        print("Data Loaded Successfully.\n")
        print(self.df.info())
        print(self.df.describe())
        print("Missing Values:\n", self.df.isnull().sum())

    def perform_eda(self):
        """
        Perform Exploratory Data Analysis with visualizations and statistics.
        """
        print("\n--- EDA ---")
        # Histograms
        self.df.hist(figsize=(12, 8))
        plt.suptitle("Feature Distributions")
        plt.tight_layout()
        plt.show()

        # Boxplots for outlier detection
        for col in self.df.select_dtypes(include=np.number).columns:
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.show()

        # Multicollinearity check using VIF
        X_num = self.df.select_dtypes(include=np.number).drop(columns=['BMI'])
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X_num.columns
        vif_data['VIF'] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
        print("\nVariance Inflation Factors:\n", vif_data)

    def preprocess_data(self):
        """
        Encode categorical variables, scale features, and split data.
        """
        print("\n--- Preprocessing ---")
        # Encode categorical variables
        self.df = pd.get_dummies(self.df, drop_first=True)

        # Feature engineering (optional)
        if 'Height' in self.df.columns and 'Weight' in self.df.columns:
            self.df['Height_to_Weight'] = self.df['Height'] / self.df['Weight']

        # Separate features and target
        self.X = self.df.drop(columns=['BMI'])
        self.y = self.df['BMI']

        # Standardize features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        print("Data Preprocessing Complete.")

    def train_models(self):
        """
        Train multiple regression models and store their performance.
        """
        print("\n--- Model Training ---")

        # Simple Linear Regression (first feature only)
        X_simple = self.X_train[:, 0].reshape(-1, 1)
        model_simple = LinearRegression()
        model_simple.fit(X_simple, self.y_train)
        y_pred_simple = model_simple.predict(self.X_test[:, 0].reshape(-1, 1))
        self.models['Simple Linear'] = model_simple
        self.results['Simple Linear'] = r2_score(self.y_test, y_pred_simple)

        # Multiple Linear Regression
        model_multi = LinearRegression()
        model_multi.fit(self.X_train, self.y_train)
        y_pred_multi = model_multi.predict(self.X_test)
        self.models['Multiple Linear'] = model_multi
        self.results['Multiple Linear'] = r2_score(self.y_test, y_pred_multi)

        # Ridge Regression
        model_ridge = Ridge(alpha=1.0)
        model_ridge.fit(self.X_train, self.y_train)
        y_pred_ridge = model_ridge.predict(self.X_test)
        self.models['Ridge'] = model_ridge
        self.results['Ridge'] = r2_score(self.y_test, y_pred_ridge)

        # Lasso Regression
        model_lasso = Lasso(alpha=0.1)
        model_lasso.fit(self.X_train, self.y_train)
        y_pred_lasso = model_lasso.predict(self.X_test)
        self.models['Lasso'] = model_lasso
        self.results['Lasso'] = r2_score(self.y_test, y_pred_lasso)

        # Elastic Net Regression
        model_elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
        model_elastic.fit(self.X_train, self.y_train)
        y_pred_elastic = model_elastic.predict(self.X_test)
        self.models['Elastic Net'] = model_elastic
        self.results['Elastic Net'] = r2_score(self.y_test, y_pred_elastic)

        print("Model Training Complete.")

    def evaluate_models(self):
        """
        Display R² scores and interpret best model.
        """
        print("\n--- Model Evaluation ---")
        for name, score in self.results.items():
            print(f"{name} R² Score: {score:.4f}")

        # Best model
        best_model_name = max(self.results, key=self.results.get)
        print(f"\nBest Model: {best_model_name}")
        best_model = self.models[best_model_name]

        # Coefficients interpretation
        if hasattr(best_model, 'coef_'):
            coef_series = pd.Series(best_model.coef_, index=self.X.columns)
            print("\nFeature Coefficients:\n", coef_series.sort_values(ascending=False))

    def polynomial_regression(self, degree=2):
        """
        Optional: Apply Polynomial Regression for non-linear relationships.
        """
        print(f"\n--- Polynomial Regression (Degree {degree}) ---")
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(self.X_scaled)
        X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
            X_poly, self.y, test_size=0.2, random_state=42
        )
        model_poly = LinearRegression()
        model_poly.fit(X_train_poly, y_train_poly)
        y_pred_poly = model_poly.predict(X_test_poly)
        score = r2_score(y_test_poly, y_pred_poly)
        print(f"Polynomial Regression R² Score: {score:.4f}")

    def cross_validation(self):
        """
        Optional: Apply cross-validation to assess generalizability.
        """
        print("\n--- Cross-Validation ---")
        model = LinearRegression()
        scores = cross_val_score(model, self.X_scaled, self.y, cv=5, scoring='r2')
        print("Cross-Validation R² Scores:", scores)
        print("Mean CV R²:", np.mean(scores))

if __name__ == '__main__':
    bmi_model = BMIPredictor('bmi_dataset.csv')
    bmi_model.load_data()
    bmi_model.perform_eda()
    bmi_model.preprocess_data()
    bmi_model.train_models()
    bmi_model.evaluate_models()
    bmi_model.polynomial_regression(degree=2)  # Optional
    bmi_model.cross_validation()