# Feature Selection and Model Evaluation Notebook

## Overview

This notebook is designed to perform feature selection, data preprocessing, and model evaluation of this data on a toxicity prediction dataset. The primary goal is to prepare the data, select relevant features.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Selection](#feature-selection)
5. [Model Training and Evaluation](#model-training-and-evaluation)

## Dataset
The dataset used in this notebook is merged_data_clean.csv, which contains combined features from PedalPy and RDKit for SMILES strings and their corresponding toxicity. The target variable is Toxicity, and the features include various chemical properties. The Source column in this dataset specifies the origin of each molecule which can be used for experementing.

## Data Preprocessing
Loading the Dataset: The dataset is loaded into a pandas DataFrame.
Filtering Data: The dataset is filtered to include only rows where the Source column is MolToxPred.
Dropping Categorical Columns: The categorical columns (SMILES and Source) are dropped from the dataset.
Handling Missing Values: Infinite and NaN values are replaced, and missing values are imputed using the mean strategy.
Standardization: The features are standardized using StandardScaler.

## Feature Selection
Duplicate Columns: Duplicate columns are identified and removed.
Boruta Algorithm: The Boruta feature selection method is applied to identify the most important features. Boruta is a wrapper method built around a random forest classifier, which adds randomness to the data and iteratively removes features that are deemed less important.
Principal Component Analysis (PCA): PCA is applied to reduce the dimensionality of the dataset and select the most important components. PCA helps in capturing the variance in the data with fewer features, making the model simpler and faster.

## Model Training and Evaluation
Some form of Model Training and Evaluation is done to check the whole process.

## Dependencies

To run this notebook, you need the following libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn 