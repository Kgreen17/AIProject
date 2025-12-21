import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from scipy.spatial.distance import cdist
from numpy.random import uniform


class IrisUnsupervisedClustering:
    """
    End-to-end Unsupervised Learning Pipeline for Iris Dataset
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.cluster_labels = None

    # ---------------------------------------------------------
    # 1. Data Loading & Initial Exploration
    # ---------------------------------------------------------
    def load_and_explore_data(self):
        self.df = pd.read_csv(self.file_path)
        print("\n--- Dataset Info ---")
        print(self.df.info())
        print("\n--- Basic Statistics ---")
        print(self.df.describe())

        # Visual distributions
        self.df.hist(figsize=(10, 6))
        plt.suptitle("Feature Distributions")
        plt.show()

        sns.pairplot(self.df.select_dtypes(include=np.number))
        plt.show()

    # ---------------------------------------------------------
    # 2. Data Cleaning & Preprocessing
    # ---------------------------------------------------------
    def preprocess_data(self):
        self.df.dropna(inplace=True)
        self.X = self.df.select_dtypes(include=np.number)
        self.X_scaled = self.scaler.fit_transform(self.X)
        print("\nData standardized successfully.")

    # ---------------------------------------------------------
    # 3. Hopkins Statistic (Cluster Tendency)
    # ---------------------------------------------------------
    def hopkins_statistic(self, sample_size=0.5):
        X = self.X_scaled
        n = X.shape[0]
        m = int(sample_size * n)

        random_indices = np.random.choice(n, m, replace=False)
        X_sample = X[random_indices]

        random_points = uniform(np.min(X, axis=0), np.max(X, axis=0), (m, X.shape[1]))

        u_dist = np.min(cdist(random_points, X), axis=1)
        w_dist = np.min(cdist(X_sample, X), axis=1)

        hopkins_value = np.sum(u_dist) / (np.sum(u_dist) + np.sum(w_dist))
        print(f"\nHopkins Statistic: {hopkins_value:.4f}")

        if hopkins_value > 0.7:
            print("✔ Data is highly clusterable")
        else:
            print("✖ Weak clustering tendency")

        return hopkins_value

    # ---------------------------------------------------------
    # 4. Determine Optimal Number of Clusters
    # ---------------------------------------------------------
    def find_optimal_k(self, max_k=10):
        wcss = []
        silhouette_scores = []

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(self.X_scaled)
            wcss.append(km.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, labels))

        # Elbow Plot
        plt.plot(range(2, max_k + 1), wcss, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("WCSS")
        plt.title("Elbow Method")
        plt.show()

        # Silhouette Plot
        plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")
        plt.show()

    # ---------------------------------------------------------
    # 5. Apply K-Means Clustering
    # ---------------------------------------------------------
    def apply_kmeans(self, k):
        self.kmeans = KMeans(n_clusters=k, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        self.df['Cluster'] = self.cluster_labels

        print("\n--- KMeans Model Summary ---")
        print(f"Clusters: {k}")
        print(f"Inertia: {self.kmeans.inertia_}")
        print(f"Iterations: {self.kmeans.n_iter_}")

    # ---------------------------------------------------------
    # 6. PCA for Visualization
    # ---------------------------------------------------------
    def apply_pca(self):
        self.pca = PCA(n_components=2)
        pca_data = self.pca.fit_transform(self.X_scaled)

        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = self.cluster_labels

        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set2')
        plt.title("PCA Cluster Visualization")
        plt.show()

    # ---------------------------------------------------------
    # 7. Interpret Cluster Centers
    # ---------------------------------------------------------
    def show_cluster_centers(self):
        centers_scaled = self.kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)

        centers_df = pd.DataFrame(centers_original, columns=self.X.columns)
        print("\n--- Cluster Centers (Original Scale) ---")
        print(centers_df)

    # ---------------------------------------------------------
    # 8. Cluster Profiling
    # ---------------------------------------------------------
    def cluster_profiling(self):
        profile = self.df.groupby('Cluster').mean()
        print("\n--- Cluster Profiling ---")
        print(profile)

        profile.plot(kind='bar', figsize=(10, 6))
        plt.title("Cluster Feature Comparison")
        plt.show()

    # ---------------------------------------------------------
    # 9. Silhouette Plot
    # ---------------------------------------------------------
    def silhouette_plot(self):
        silhouette_vals = silhouette_samples(self.X_scaled, self.cluster_labels)

        y_lower = 10
        for i in np.unique(self.cluster_labels):
            ith_cluster_vals = silhouette_vals[self.cluster_labels == i]
            ith_cluster_vals.sort()
            size_cluster = ith_cluster_vals.shape[0]

            y_upper = y_lower + size_cluster
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_vals)
            y_lower = y_upper + 10

        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.title("Silhouette Plot")
        plt.show()

    # ---------------------------------------------------------
    # 10. Conclusion Helper
    # ---------------------------------------------------------
    def conclusion(self):
        print("""
        ✔ Successfully clustered Iris dataset using K-Means.
        ✔ PCA helped visualize clear cluster separation.
        ✔ Clusters align well with flower measurements.
        ✔ Future work: Try DBSCAN or Hierarchical Clustering.
        """)
