class BMIDeepLearningCapstone:
    def __init__(self, csv_path, target_column="BMI"):
        self.csv_path = csv_path
        self.target_column = target_column

        # Data containers
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Preprocessing
        self.scaler = None

        # Model
        self.model = None
        self.history = None

    # ----------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------
    def load_data(self):
        import pandas as pd
        self.df = pd.read_csv(self.csv_path)
        print("Dataset Loaded Successfully")
        print(self.df.head())

    # ----------------------------------------------------
    # 2. Exploratory Data Analysis (EDA)
    # ----------------------------------------------------
    def perform_eda(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("\nDataset Info:")
        print(self.df.info())

        print("\nStatistical Summary:")
        print(self.df.describe())

        # Correlation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()

        # Target distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(self.df[self.target_column], kde=True)
        plt.title("BMI Distribution")
        plt.show()

    # ----------------------------------------------------
    # 3. Preprocessing
    # ----------------------------------------------------
    def preprocess_data(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        # Separate features & target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        # Encode categorical variables if any
        self.X = pd.get_dummies(self.X, drop_first=True)

        # Train-Test Split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Feature Scaling
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("Preprocessing Completed")

    # ----------------------------------------------------
    # 4. Build ANN Model
    # ----------------------------------------------------
    def build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        self.model = Sequential([
            Dense(64, activation="relu", input_shape=(self.X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1)  # Regression output
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        print("ANN Model Built Successfully")
        self.model.summary()

    # ----------------------------------------------------
    # 5. Train Model
    # ----------------------------------------------------
    def train_model(self, epochs=100, batch_size=16):
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    # ----------------------------------------------------
    # 6. Evaluate Model
    # ----------------------------------------------------
    def evaluate_model(self):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np

        y_pred = self.model.predict(self.X_test).flatten()

        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        print("\nModel Evaluation Metrics:")
        print(f"R² Score : {r2:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"MSE      : {mse:.4f}")
        print(f"RMSE     : {rmse:.4f}")

        return y_pred

    # ----------------------------------------------------
    # 7. Visualization
    # ----------------------------------------------------
    def visualize_results(self, y_pred):
        import matplotlib.pyplot as plt

        # Actual vs Predicted
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 "r--")
        plt.xlabel("Actual BMI")
        plt.ylabel("Predicted BMI")
        plt.title("Actual vs Predicted BMI")
        plt.show()

        # Training vs Validation Loss
        plt.figure(figsize=(6, 4))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()

    # ----------------------------------------------------
    # 8. Save Model & Scaler (Optional)
    # ----------------------------------------------------
    def save_artifacts(self):
        import pickle

        self.model.save("bmi_ann_model.keras")

        with open("bmi_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print("Model and Scaler Saved Successfully")

