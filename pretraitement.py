import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Chargement des données
obesity_data = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv') 

# Sélection des colonnes pour One-Hot Encoding
categorical_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
obesity_dummies = pd.get_dummies(obesity_data[categorical_features])

# Sélection des colonnes numériques
obesity_numeric = obesity_data.select_dtypes(include=["int64", "float64"])

# Ajout des étiquettes de la variable cible
obesity_lab = obesity_data[["NObeyesdad"]]

# Concatenation des données prétraitées
obesity_concatenated = pd.concat([obesity_numeric, obesity_dummies, obesity_lab], axis=1)

# Séparation des caractéristiques et des étiquettes
obesity_features = obesity_concatenated.drop("NObeyesdad", axis=1).astype("float")
obesity_label = obesity_concatenated["NObeyesdad"]

# Normalisation des caractéristiques
scaler = MinMaxScaler()
obesity_features_scaled = scaler.fit_transform(obesity_features)

# Encodage des étiquettes
encoder = LabelEncoder()
obesity_labels_encoded = encoder.fit_transform(obesity_label)

# Division des données en ensembles d'entraînement et de test
obesity_features_tr, obesity_features_ts, obesity_labels_tr, obesity_labels_ts = train_test_split(
    obesity_features_scaled, obesity_labels_encoded,
    test_size=0.2, stratify=obesity_labels_encoded,
    random_state=42
)

# Sauvegarde des ensembles sous forme de fichiers CSV si nécessaire
pd.DataFrame(obesity_features_tr).to_csv("obesity_features_train.csv", index=False)
pd.DataFrame(obesity_features_ts).to_csv("obesity_features_test.csv", index=False)
pd.DataFrame(obesity_labels_tr).to_csv("obesity_labels_train.csv", index=False)
pd.DataFrame(obesity_labels_ts).to_csv("obesity_labels_test.csv", index=False)

print("Prétraitement terminé et fichiers sauvegardés.")
