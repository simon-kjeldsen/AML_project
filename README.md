# AML Project – Danish Housing Prices

This repository contains the code for an Applied Machine Learning project focused on predicting Danish housing prices using structured tabular data. Multiple models are implemented and evaluated, including a simple baseline, Linear Regression, K-Nearest Neighbours (KNN), and Random Forest. Model performance is assessed using test-set metrics and cross-validation.

## Setup

The code was executed in a standard Python environment. Required packages are listed in `requirements.txt`.

## Data 

The dataset used in this project is located in the data/ folder:

data/DKHousingPricesSample100k.csv

All scripts expect the data path to be passed explicitly via a --data argument.

## Models

All scripts are run from the project root.

### Baseline model

Train baseline model and save it to the models/ folder:

python src/train_baselinemodel.py --data data/DKHousingPricesSample100k.csv

Run cross-validation for the baseline model:

python src/cv_baselinemodel.py --data data/DKHousingPricesSample100k.csv

### Linear Regression

Train Linear Regression model:

python src/train_linearreg.py --data data/DKHousingPricesSample100k.csv


Run cross-validation for Linear Regression:

python src/cv_linearreg.py --data data/DKHousingPricesSample100k.csv


### K-Nearest Neighbours (KNN)

Train KNN model:

python src/train_knn.py --data data/DKHousingPricesSample100k.csv


Run cross-validation for KNN:

python src/cv_knn.py --data data/DKHousingPricesSample100k.csv

### Random Forest

Train Random Forest model:

python src/train_rf.py --data data/DKHousingPricesSample100k.csv


Run cross-validation for Random Forest:

python src/cv_rf.py --data data/DKHousingPricesSample100k.csv


## Results and evaluation

Test-set performance metrics (MAE, RMSE, R²) are printed to the console when running the training scripts.

Cross-validated R² is reported when running the cv_*.py scripts.

Trained models are saved in the models/ folder as .joblib files.

Any generated figures or summaries can be stored in the reports/ folder.

## Utilities

Common helper functions for data loading, cleaning, and preprocessing are defined in:

src/utils.py

These utilities are shared across all training and evaluation scripts.


