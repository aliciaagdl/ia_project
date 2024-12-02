import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def train_kmeans(features, n_clusters=7):
    """
    Function: Trains a KMeans clustering model.
    
    Parameters:
        features: Feature set for clustering.
        n_clusters: Number of clusters (default=7).

    Returns:
        Trained KMeans model and predicted labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    predicted_labels = kmeans.fit_predict(features)
    
    return kmeans, predicted_labels

if __name__ == "__main__":
    # Charger les données (exemple)
    obesity_features_ts = pd.read_csv("obesity_features_test.csv").values
    
    kmeans_model, predicted_labels = train_kmeans(obesity_features_ts)
    
    # Sauvegarde des étiquettes prédites
    np.savetxt("predicted_labels.csv", predicted_labels, delimiter=",")
