import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def count_values(dataset, cat_variable, order=None):
    """
    Function: Counts values in each category and displays them on a plot.
    
    Parameters: Dataset, category feature, and order of appearance (order is optional).
    """
    ax = sns.countplot(x=cat_variable, data=dataset, palette="Blues_r", order=order)
    for p in ax.patches:
        ax.annotate(f"\n{p.get_height()}", (p.get_x() + 0.2, p.get_height()),
                    ha="center", va="top", color="white", size=10)
    
    plt.title(f"Number of items in each {cat_variable} category")
    plt.savefig(f"./reports/figures/Number of items in each {cat_variable} category")
    plt.show()

def plot_distribution(dataset, feature):
    """
    Function: Computes and displays distribution of features with continuous values; plots their mean and median.
    
    Parameters: Dataset and feature with continuous values.
    """
    plt.hist(dataset[feature], bins="fd", alpha=0.7, color='skyblue')
    plt.axvline(dataset[feature].mean(), color="red", label="Mean")
    plt.axvline(dataset[feature].median(), color="orange", label="Median")
    
    plt.xlabel(f"{feature}")
    plt.ylabel("Count")
    plt.legend()
    plt.title(f"Distribution of values in {feature}")
    plt.savefig(f"./reports/figures/Distribution of values in {feature}")
    plt.show()

def plot_correlation(dataset, feature_x, feature_y):
    """
    Function: Plots the correlation between two continuous variables.
    """
    plt.scatter(dataset[feature_x], dataset[feature_y], alpha=0.5)
    m, b = np.polyfit(dataset[feature_x], dataset[feature_y], 1)
    plt.plot(dataset[feature_x], m * dataset[feature_x] + b, color="red")

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"Correlation between '{feature_x}' and '{feature_y}'")
    plt.savefig(f"./reports/figures/Correlation between '{feature_x}' and '{feature_y}'")
    plt.show()
    
    r = np.corrcoef(dataset[feature_x], dataset[feature_y])[0, 1]
    print(f"Correlation coefficient (r): {r}")

def cross_plot(dataset, lead_category, sup_category, order=None):
    """
    Function: Plots interaction between two categorical variables.
    
    Parameters: Dataset, lead category, supplemental category, and order of appearance (optional).
    """
    sns.countplot(x=lead_category, hue=sup_category, data=dataset, order=order, palette="Blues_r")
    plt.title(f"Interaction between {lead_category} and {sup_category}")
    plt.savefig(f"./reports/figures/Interaction between {lead_category} and {sup_category}")
    plt.show()

if __name__ == "__main__":
    # Load dataset
    obesity_data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    
    # Examples of visualizations
    count_values(obesity_data, "Gender")
    plot_distribution(obesity_data, "Age")
    plot_distribution(obesity_data, "Height")
    plot_distribution(obesity_data, "Weight")
    plot_correlation(obesity_data, "Height", "Weight")
    cross_plot(obesity_data, "NObeyesdad", "Gender", order=[
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
        "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"])