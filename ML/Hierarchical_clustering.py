class IrisHierarchicalClustering:
    """
    Assignment: Hierarchical Clustering on the Iris Dataset
    """

    def __init__(self, n_clusters=3, linkage_method="ward"):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.df = None
        self.features_scaled = None
        self.linkage_matrix = None
        self.cluster_labels = None

    # -----------------------------------------------------
    # 1. Load Dataset
    # -----------------------------------------------------
    def load_data(self):
        from sklearn.datasets import load_iris
        import pandas as pd

        iris = load_iris()
        self.df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )
        self.df["species"] = iris.target
        self.df["species_name"] = iris.target_names[iris.target]

        print("First 5 rows of the dataset:")
        display(self.df.head())

    # -----------------------------------------------------
    # 2. Data Preprocessing (Normalization)
    # -----------------------------------------------------
    def preprocess_data(self):
        from sklearn.preprocessing import StandardScaler

        X = self.df.drop(columns=["species", "species_name"])
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(X)

        print(
            "Features normalized using StandardScaler.\n"
            "Normalization is important because hierarchical clustering\n"
            "is distance-based and features must contribute equally."
        )

    # -----------------------------------------------------
    # 3. Create Linkage Matrix
    # -----------------------------------------------------
    def create_linkage_matrix(self):
        from scipy.cluster.hierarchy import linkage

        self.linkage_matrix = linkage(
            self.features_scaled,
            method=self.linkage_method
        )

        print(f"Linkage matrix created using '{self.linkage_method}' method.")

    # -----------------------------------------------------
    # 4. Plot Dendrogram
    # -----------------------------------------------------
    def plot_dendrogram(self, cut_height=7):
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        plt.figure(figsize=(14, 6))
        dendrogram(self.linkage_matrix)
        plt.axhline(y=cut_height, color="red", linestyle="--")
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.show()

        print(
            f"Dendrogram plotted with horizontal cut at height = {cut_height}.\n"
            "This cut typically results in 3 clusters."
        )

    # -----------------------------------------------------
    # 5. Assign Cluster Labels
    # -----------------------------------------------------
    def assign_clusters(self):
        from scipy.cluster.hierarchy import fcluster

        self.cluster_labels = fcluster(
            self.linkage_matrix,
            self.n_clusters,
            criterion="maxclust"
        )
        self.df["cluster"] = self.cluster_labels

        print("Cluster labels assigned using fcluster().")
        display(self.df.head())

    # -----------------------------------------------------
    # 6. Compare with Actual Species
    # -----------------------------------------------------
    def compare_with_species(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import pandas as pd

        cm = confusion_matrix(self.df["species"], self.df["cluster"])

        cm_df = pd.DataFrame(
            cm,
            index=["Setosa", "Versicolor", "Virginica"],
            columns=[f"Cluster {i}" for i in range(1, self.n_clusters + 1)]
        )

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix: Species vs Clusters")
        plt.ylabel("Actual Species")
        plt.xlabel("Cluster Label")
        plt.show()

        print("Confusion matrix plotted to compare clusters with true species.")

    # -----------------------------------------------------
    # 7. Optional: PCA Visualization
    # -----------------------------------------------------
    def pca_visualization(self):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.features_scaled)

        pca_df = pd.DataFrame(
            pca_data,
            columns=["PC1", "PC2"]
        )
        pca_df["Cluster"] = self.df["cluster"]
        pca_df["Species"] = self.df["species_name"]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="Cluster",
            palette="Set1"
        )
        plt.title("PCA - Cluster Visualization")

        plt.subplot(1, 2, 2)
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="Species",
            palette="Set2"
        )
        plt.title("PCA - Actual Species")

        plt.tight_layout()
        plt.show()

        print("PCA visualization completed.")

    # -----------------------------------------------------
    # Run Everything
    # -----------------------------------------------------
    def run_all(self):
        self.load_data()
        self.preprocess_data()
        self.create_linkage_matrix()
        self.plot_dendrogram()
        self.assign_clusters()
        self.compare_with_species()
        self.pca_visualization()
