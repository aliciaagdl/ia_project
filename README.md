# IA_Project


# README.md

## Multi-Class Classification of Obesity Levels Based on Eating Habits and Physical Conditions

### Project Overview

This project focuses on the multi-class classification of obesity levels using a dataset collected from individuals in Colombia, Peru, and Mexico. The dataset contains detailed information about eating habits, physical activities, and sociodemographic characteristics. The goal is to develop predictive models that classify individuals into one of seven obesity categories:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

### Dataset Description

The dataset used in this project is publicly available under a Creative Commons license. It consists of **2111 records** and **17 attributes**. Data is divided as follows:

- **23%**: Collected directly from users via a web platform.
- **77%**: Synthetically generated using the Weka tool and the SMOTE filter for balancing.

**Attributes:**
- Eating habits: 
  - Frequent consumption of high caloric food (FAVC)
  - Frequency of consumption of vegetables (FCVC)
  - Number of main meals (NCP)
  - Consumption of food between meals (CAEC)
  - Daily water consumption (CH2O)
  - Alcohol consumption (CALC)
- Physical conditions:
  - Calories consumption monitoring (SCC)
  - Physical activity frequency (FAF)
  - Time using technology devices (TUE)
  - Transportation method (MTRANS)
- Demographic and physical attributes: Gender, Age, Height, Weight
- Target variable: **NObesity** (Obesity Level)

**Data Formats:** Available in CSV and ARFF for Weka compatibility.

### Methodology

The project employs a variety of machine learning techniques to analyze and classify obesity levels:

1. **Exploratory Data Analysis (EDA):**
   - Visualization of data distributions.
   - Investigation of correlations between physical attributes and obesity levels.

2. **Preprocessing:**
   - Handling missing values.
   - Feature scaling using MinMaxScaler.
   - Encoding categorical variables using One-Hot Encoding and Label Encoding.

3. **Model Development:**
   - **Decision Tree Classifier:** Optimized using GridSearchCV to determine the best hyperparameters.
   - **Clustering Analysis:** Utilized KMeans for visualizing clusters of obesity levels.

4. **Evaluation Metrics:**
   - Accuracy
   - F1-Score (weighted)
   - Confusion Matrix
   - Classification Report

### Key Findings

- Achieved satisfactory classification performance using a Decision Tree model.
- Visualized relationships between features like height, weight, and obesity levels.
- Explored clustering patterns to understand groupings of individuals based on physical attributes.

### Project Requirements

- **Programming Language:** Python
- **Libraries:** 
  - Data Processing: `numpy`, `pandas`
  - Visualization: `matplotlib`, `seaborn`, `scikit-plot`
  - Machine Learning: `scikit-learn`



### Results Visualization

- Feature importance in decision tree classification.
- Heatmaps for confusion matrices.
- Cluster plots using KMeans.

### Dataset Citation

Mendoza Palechor, F., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru, and Mexico. Available under [Creative Commons License](https://doi.org/10.1016/j.dib.2019.104344).



**Author:** AGOUDJIL Alicia
**Date:** December 2024

## Project Organization

```
├── LICENSE            
├── Makefile           
├── README.md          
├── data
│   ├── external
│   ├── interim        
│   ├── processed      
│   └── raw    
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── docs               
│
├── models  
│     ├── decision_tree.pkl
├── notebooks          
│                         
│                         
│
├── pyproject.toml     
│                        
│
├── references         
│
├── reports           
│   └── figures        
│
├── requirements.txt   
│                        
│
├── setup.cfg          
│
└── ia  
    │
    ├── __init__.py             
    │
    ├── config.py               
    │
    ├── dataset.py              
    │
    ├── features.py             
    │
    ├── modeling                
    │   ├── main.py 
    │   ├── predict.py                  
    │   └── train.py            
    │   └── test.py 
    │   └── train_kmeans
    │   └── viz_data.py
    │   └── plot_clusters.py
    │   └── pretraitement.py
    │
    └── plots.py                
```

--------

