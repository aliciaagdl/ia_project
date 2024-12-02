import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def train_model(train_features, train_labels):
    """
    Function: Trains a DecisionTreeClassifier using GridSearchCV.
    
    Parameters:
        train_features: Training feature set.
        train_labels: Training labels.

    Returns:
        Best model after hyperparameter tuning.
    """
    f1 = make_scorer(f1_score, average="weighted")
    params = {"max_depth": [5, 7, 9, 11, 13, 15]}
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=params, cv=5, scoring=f1)
    
    grid_search.fit(train_features, train_labels)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    return best_model

if __name__ == "__main__":
    # Charger les données (exemple)
    obesity_features_tr = pd.read_csv("obesity_features_train.csv").values
    obesity_labels_tr = pd.read_csv("obesity_labels_train.csv").values.ravel()
    
    model_tree = train_model(obesity_features_tr, obesity_labels_tr)
