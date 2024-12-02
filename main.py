import pandas as pd
import numpy as np
import pickle
import pretraitement as pt
import viz_data as viz
from Train import train_model
from predict import predict_and_evaluate
from train_kmeans import train_kmeans
from plot_clusters import plot_clusters

def main():
    
    # 1. Count plot pour les catégories
    print("Distribution par genre:")
    viz.count_values(pt.obesity_data, "Gender")
    
    print("Historique familial de surpoids:")
    viz.count_values(pt.obesity_data, "family_history_with_overweight")
    
    print("Consommation fréquente de nourriture à haute teneur calorique:")
    viz.count_values(pt.obesity_data, "FAVC")
    
    print("Fréquence des repas:")
    viz.count_values(pt.obesity_data, "CAEC", ["no", "Sometimes", "Frequently", "Always"])
    
    print("Habitudes tabagiques:")
    viz.count_values(pt.obesity_data, "SMOKE")
    
    print("Consommation d'alcool:")
    viz.count_values(pt.obesity_data, "CALC")
    
    print("Modes de transport:")
    viz.count_values(pt.obesity_data, "MTRANS")
    
    print("Niveaux d'obésité:")
    viz.count_values(pt.obesity_data, "NObeyesdad", [
        "Insufficient_Weight", "Normal_Weight", 
        "Overweight_Level_I", "Overweight_Level_II", 
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ])
    
    # 2. Distributions des variables continues
    print("Distribution de l'âge:")
    viz.plot_distribution(pt.obesity_data, "Age")
    
    print("Distribution de la taille:")
    viz.plot_distribution(pt.obesity_data, "Height")
    
    print("Distribution du poids:")
    viz.plot_distribution(pt.obesity_data, "Weight")
    
    # 3. Corrélation entre poids et taille
    print("Corrélation entre taille et poids:")
    viz.plot_correlation(pt.obesity_data, "Height", "Weight")
    
    # 4. Visualisation des interactions catégoriques
    print("Interaction entre niveaux d'obésité et genre:")
    viz.cross_plot(pt.obesity_data, "NObeyesdad", "Gender", order=[
        "Insufficient_Weight", "Normal_Weight", 
        "Overweight_Level_I", "Overweight_Level_II", 
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ])
    
    print("Interaction entre niveaux d'obésité et consommation d'alcool:")
    viz.cross_plot(pt.obesity_data, "NObeyesdad", "CALC", order=[
        "Insufficient_Weight", "Normal_Weight", 
        "Overweight_Level_I", "Overweight_Level_II", 
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ])
    
    print("Interaction entre niveaux d'obésité et fréquence des repas:")
    viz.cross_plot(pt.obesity_data, "NObeyesdad", "CAEC", order=[
        "Insufficient_Weight", "Normal_Weight", 
        "Overweight_Level_I", "Overweight_Level_II", 
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ])
    
    print("Interaction entre niveaux d'obésité et modes de transport:")
    viz.cross_plot(pt.obesity_data, "NObeyesdad", "MTRANS", order=[
        "Insufficient_Weight", "Normal_Weight", 
        "Overweight_Level_I", "Overweight_Level_II", 
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ])

    # Étape 1: Entraîner le modèle de classification
    print("==== Entraînement du modèle de classification ====")
    model_tree = train_model(pt.obesity_features_tr, pt.obesity_labels_tr)
    with open('./models/decision_tree.pkl', 'wb') as f:
        pickle.dump(model_tree, f)
    
    # Étape 2: Prédiction et évaluation
    print("\n==== Évaluation du modèle de classification ====")
    predict_and_evaluate(model_tree, pt.obesity_features_tr, pt.obesity_labels_tr, pt.obesity_features_ts, pt.obesity_labels_ts)
    
    # Étape 3: Entraînement KMeans pour le clustering
    print("\n==== Entraînement KMeans pour le clustering ====")
    kmeans_model, predicted_labels = train_kmeans(pt.obesity_features_ts)
    
    # Sauvegarde des étiquettes prédites
    pd.DataFrame(predicted_labels, columns=["Cluster_Label"]).to_csv("predicted_labels.csv", index=False)
    
    # Étape 4: Visualisation des clusters réels et prédits
    print("\n==== Visualisation des clusters ====")
    
    # Charger les données test pour la visualisation
    obesity_features_ts_df = pd.DataFrame(pt.obesity_features_ts, columns=pt.obesity_features.columns)
    
    # Clusters réels (labels test)
    plot_clusters(obesity_features_ts_df, "Height", "Weight", pt.obesity_labels_ts, "Clusters réels (Test)")
    
    # Clusters prédits
    plot_clusters(obesity_features_ts_df, "Height", "Weight", predicted_labels, "Clusters prédits (KMeans)")

if __name__ == "__main__":
    main()
