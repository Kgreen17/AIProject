import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


class OnlineRetailClustering:
    """
    End-to-end implementation of:
    - Data Cleaning
    - RFM Feature Engineering
    - PCA Dimensionality Reduction
    - K-Means Clustering
    - Hierarchical Clustering
    - Visualization & Interpretation
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.rfm_df = None
        self.scaled_rfm = None
        self.pca_data = None
        self.pca = None

    # ---------------------------------------------------------
    # Step 1: Data Cleaning & Preprocessing
    # ---------------------------------------------------------
    def load_and_clean_data(self):
        print("Loading dataset...")
        self.df = pd.read_excel(self.file_path)

        print("Cleaning data...")
        self.df = self.df[
            (self.df["Quantity"] > 0) &
            (~self.df["InvoiceNo"].astype(str).str.startswith("C"))
        ]

        self.df.dropna(subset=["CustomerID"], inplace=True)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        print(f"Cleaned dataset shape: {self.df.shape}")

    # ---------------------------------------------------------
    # Step 2: RFM Feature Engineering
    # ---------------------------------------------------------
    def create_rfm_features(self):
        print("Creating RFM features...")

        snapshot_date = self.df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = self.df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        })

        rfm.columns = ["Recency", "Frequency", "Monetary"]
        self.rfm_df = rfm

        print("RFM feature creation completed.")
        return self.rfm_df.head()

    # ---------------------------------------------------------
    # Step 3: Scaling & PCA
    # ---------------------------------------------------------
    def scale_and_apply_pca(self, n_components=2):
        print("Scaling RFM features...")
        scaler = StandardScaler()
        self.scaled_rfm = scaler.fit_transform(self.rfm_df)

        print("Applying PCA...")
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.scaled_rfm)

        print("Explained Variance Ratio:", self.pca.explained_variance_ratio_)
        return self.pca_data

    def plot_explained_variance(self):
        plt.figure()
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.show()

    def plot_pca_2d(self):
        plt.figure()
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], s=10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA - First Two Components")
        plt.show()

    # ---------------------------------------------------------
    # Step 4: K-Means Clustering
    # ---------------------------------------------------------
    def elbow_method(self, max_k=10):
        print("Running Elbow Method...")
        distortions = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.pca_data)
            distortions.append(kmeans.inertia_)

        plt.figure()
        plt.plot(range(2, max_k + 1), distortions, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for K-Means")
        plt.show()

    def apply_kmeans(self, n_clusters=4):
        print(f"Applying K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.rfm_df["KMeansCluster"] = kmeans.fit_predict(self.pca_data)
        return self.rfm_df["KMeansCluster"].value_counts()

    def plot_kmeans_clusters(self):
        plt.figure()
        for cluster in self.rfm_df["KMeansCluster"].unique():
            mask = self.rfm_df["KMeansCluster"] == cluster
            plt.scatter(
                self.pca_data[mask, 0],
                self.pca_data[mask, 1],
                s=10,
                label=f"Cluster {cluster}"
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("K-Means Clustering on PCA Data")
        plt.legend()
        plt.show()

    def interpret_kmeans_clusters(self):
        print("K-Means Cluster Profiles:")
        return self.rfm_df.groupby("KMeansCluster").mean()

    # ---------------------------------------------------------
    # Step 5: Hierarchical Clustering
    # ---------------------------------------------------------
    def plot_dendrogram(self, sample_size=1000):
        print("Plotting dendrogram (sampled data)...")
        sample = self.scaled_rfm[:sample_size]
        linked = linkage(sample, method="ward")

        plt.figure(figsize=(10, 5))
        dendrogram(linked, truncate_mode="level", p=5)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.show()

    def apply_hierarchical_clustering(self, n_clusters=4):
        print(f"Applying Hierarchical Clustering with {n_clusters} clusters...")
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        self.rfm_df["HierarchicalCluster"] = hc.fit_predict(self.scaled_rfm)
        return self.rfm_df["HierarchicalCluster"].value_counts()

    # ---------------------------------------------------------
    # Step 6: Final Analysis Helpers
    # ---------------------------------------------------------
    def compare_clusterings(self):
        print("Comparing K-Means vs Hierarchical Clustering...")
        comparison = self.rfm_df.groupby(
            ["KMeansCluster", "HierarchicalCluster"]
        ).size()
        return comparison
