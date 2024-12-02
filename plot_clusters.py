import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_clusters(dataset, feature_one, feature_two, labels, title=None):
    """
    Function: Computes and displays clusters.
    
    Parameters:
        dataset: Dataframe with features.
        feature_one, feature_two: Columns for x and y axes.
        labels: Cluster labels for each point.
        title: Title of the plot (default=None).
    """
    sns.scatterplot(data=dataset, x=feature_one, y=feature_two, hue=labels, palette="Blues_r", legend='full')
    if title is not None:
        plt.title(title)
    plt.savefig(f"./reports/figures/{title}")
    plt.show()

if __name__ == "__main__":
    # Charger les données (exemple)
    obesity_features_ts = pd.read_csv("obesity_features_test.csv")
    
    # Charger les étiquettes prédites
    predicted_labels = pd.read_csv("predicted_labels.csv", header=None).squeeze()
    
    # Plot des clusters
    plot_clusters(obesity_features_ts, "Height", "Weight", predicted_labels, "Predicted Clusters")
