import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import Train

def predict_and_evaluate(estimator, train_features, train_labels, test_features, test_labels):
    """
    Function: Predicts classes and evaluates model performance.
    
    Parameters:
        estimator: Trained model.
        train_features, train_labels: Training data.
        test_features, test_labels: Test data.
    """
    print("Training Evaluation:")
    print(f"Accuracy on Train data: {accuracy_score(train_labels, estimator.predict(train_features))}")
    print(f"F1 score on Train data: {f1_score(train_labels, estimator.predict(train_features), average='weighted')}")
    
    print("\nTest Evaluation:")
    print(f"Accuracy on Test data: {accuracy_score(test_labels, estimator.predict(test_features))}")
    print(f"F1 score on Test data: {f1_score(test_labels, estimator.predict(test_features), average='weighted')}")
    
    print("\nClassification Report on Test data:")
    print(classification_report(test_labels, estimator.predict(test_features)))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(test_labels, estimator.predict(test_features)),
                annot=True, fmt=".0f", cmap="Blues_r", linewidths=2, linecolor="white",
                xticklabels=estimator.classes_, yticklabels=estimator.classes_)
    plt.title("Confusion Matrix")
    plt.savefig("./reports/figures/Confusion_Matrix")
    plt.show()
    
    # Decision Tree Plot
    plt.figure(figsize=(22, 6))
    plot_tree(estimator, max_depth=2, feature_names=None, filled=True)
    plt.title("Decision Tree (Max Depth = 2)")
    plt.savefig("./reports/figures/Decision_Tree_(Max Depth = 2)")
    plt.show()

if __name__ == "__main__":
    # Charger les données (exemple)
    obesity_features_tr = pd.read_csv("obesity_features_train.csv").values
    obesity_labels_tr = pd.read_csv("obesity_labels_train.csv").values.ravel()
    obesity_features_ts = pd.read_csv("obesity_features_test.csv").values
    obesity_labels_ts = pd.read_csv("obesity_labels_test.csv").values.ravel()
    
    # Charger le modèle sauvegardé ou réutiliser le modèle de l'entraînement
    
    model_tree = Train.train_model(obesity_features_tr, obesity_labels_tr)
    
    predict_and_evaluate(model_tree, obesity_features_tr, obesity_labels_tr, obesity_features_ts, obesity_labels_ts)
