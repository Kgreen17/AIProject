import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer

import joblib


class CarPricePredictor:

    def __init__(self, file_path, target="Selling_Price"):
        self.file_path = file_path
        self.target = target
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.models = {}
        self.results = {}
        self.best_dt_model = None
        self.feature_names = None

    # -------------------------------------------------
    # 1. Load & Clean Data
    # -------------------------------------------------
    def load_and_clean(self):
        print("\n📌 Loading dataset...")
        self.df = pd.read_csv(self.file_path)

        print("Initial shape:", self.df.shape)
        self.df.drop_duplicates(inplace=True)

        # Convert Kms_Driven to numeric if needed
        if "Kms_Driven" in self.df.columns and self.df["Kms_Driven"].dtype == object:
            self.df["Kms_Driven"] = (
                self.df["Kms_Driven"]
                .str.replace(",", "")
                .str.extract(r"(\d+)")
                .astype(float)
            )

        # Create Car Age if Year column exists
        if "Year" in self.df.columns:
            self.df["Car_Age"] = 2025 - self.df["Year"]

        # Group Car_Name to reduce cardinality
        if "Car_Name" in self.df.columns:
            top_n = 10
            top_names = self.df["Car_Name"].value_counts().nlargest(top_n).index
            self.df["Car_Name_grouped"] = self.df["Car_Name"].where(
                self.df["Car_Name"].isin(top_names), "Other"
            )

        print("Cleaned shape:", self.df.shape)
        return self.df

    # -------------------------------------------------
    # 2. EDA Plots
    # -------------------------------------------------
    def run_eda(self):

        print("\n📊 Running EDA...")

        # 1) Histogram
        plt.figure(figsize=(7, 5))
        plt.hist(self.df[self.target], bins=30)
        plt.title("Selling Price Distribution")
        plt.show()

        # 2) Avg Price by Car Name
        name_col = (
            "Car_Name_grouped"
            if "Car_Name_grouped" in self.df.columns
            else "Car_Name"
        )

        plt.figure(figsize=(12, 6))
        self.df.groupby(name_col)[self.target].mean().sort_values().plot(kind="bar")
        plt.title("Average Selling Price by Car Name")
        plt.show()

        # 3) Scatter: Kms vs Price
        if "Kms_Driven" in self.df.columns:
            plt.figure(figsize=(7, 5))
            plt.scatter(self.df["Kms_Driven"], self.df[self.target], alpha=0.4)
            plt.xlabel("Kms Driven")
            plt.ylabel("Selling Price")
            plt.title("Kms Driven vs Selling Price")
            plt.show()

        # 4) Boxplots
        for col in ["Fuel_Type", "Transmission"]:
            if col in self.df.columns:
                plt.figure(figsize=(7, 5))
                sns.boxplot(x=self.df[col], y=self.df[self.target])
                plt.title(f"Selling Price by {col}")
                plt.show()

        # 5) Heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(self.df.select_dtypes(include="number").corr(), annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

    # -------------------------------------------------
    # 3. Preprocessing + Train/Test Split
    # -------------------------------------------------
    def preprocess_and_split(self):

        print("\n🔧 Preprocessing...")

        X = self.df.drop(columns=[self.target], errors="ignore")
        y = self.df[self.target].values

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # Preprocessor pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
                ]), cat_cols),
            ]
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # -------------------------------------------------
    # 4. Evaluate Model Helper
    # -------------------------------------------------
    @staticmethod
    def evaluate(model, X_test, y_test):
        preds = model.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
            "preds": preds,
        }

    # -------------------------------------------------
    # 5. Train Base Models
    # -------------------------------------------------
    def train_models(self):

        print("\n🤖 Training Models...")

        self.models = {
            "DecisionTree": Pipeline([
                ("pre", self.preprocessor),
                ("model", DecisionTreeRegressor(random_state=42)),
            ]),
            "LinearRegression": Pipeline([
                ("pre", self.preprocessor),
                ("model", LinearRegression()),
            ]),
            "SVR": Pipeline([
                ("pre", self.preprocessor),
                ("model", SVR()),
            ]),
            "KNN": Pipeline([
                ("pre", self.preprocessor),
                ("model", KNeighborsRegressor()),
            ]),
        }

        # Train and evaluate all
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.results[name] = self.evaluate(model, self.X_test, self.y_test)

        return self.results

    # -------------------------------------------------
    # 6. Decision Tree Tuning with GridSearchCV
    # -------------------------------------------------
    def tune_decision_tree(self):

        print("\n🔍 GridSearchCV Tuning Decision Tree...")

        pipe = Pipeline([
            ("pre", self.preprocessor),
            ("model", DecisionTreeRegressor(random_state=42)),
        ])

        params = {
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }

        grid = GridSearchCV(pipe, params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        print("Best Params:", grid.best_params_)

        self.best_dt_model = grid.best_estimator_
        self.results["DecisionTree_Tuned"] = self.evaluate(
            self.best_dt_model, self.X_test, self.y_test
        )

        return self.results["DecisionTree_Tuned"]

    # -------------------------------------------------
    # 7. Feature Importance (Decision Tree)
    # -------------------------------------------------
    def plot_feature_importance(self):

        print("\n📌 Feature Importance (Decision Tree)...")

        # Extract final feature names
        pre = self.best_dt_model.named_steps["pre"]
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][1]["onehot"].get_feature_names_out(
            pre.transformers_[1][2]
        )

        self.feature_names = list(num_cols) + list(cat_cols)

        importances = self.best_dt_model.named_steps["model"].feature_importances_

        fi = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        fi.head(20).plot(kind="bar")
        plt.title("Top 20 Feature Importances")
        plt.show()

    # -------------------------------------------------
    # 8. Actual vs Predicted Plot
    # -------------------------------------------------
    def plot_actual_vs_predicted(self):

        preds = self.results["DecisionTree_Tuned"]["preds"]

        plt.figure(figsize=(7, 6))
        plt.scatter(self.y_test, preds, alpha=0.5)
        plt.plot([min(self.y_test), max(self.y_test)],
                 [min(self.y_test), max(self.y_test)],
                 "k--")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted (Decision Tree Tuned)")
        plt.show()

    # -------------------------------------------------
    # 9. Save Best Model
    # -------------------------------------------------
    def save_model(self, filename="best_dt_model.joblib"):
        joblib.dump(self.best_dt_model, filename)
        print("\n💾 Model saved as:", filename)


# =======================================================
# HOW TO RUN
# =======================================================
if __name__ == "__main__":

    predictor = CarPricePredictor("car_data.csv")

    predictor.load_and_clean()
    predictor.run_eda()
    predictor.preprocess_and_split()
    predictor.train_models()
    predictor.tune_decision_tree()
    predictor.plot_feature_importance()
    predictor.plot_actual_vs_predicted()
    predictor.save_model()
