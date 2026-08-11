# Walmart Sales Forecasting Project

## About the Project

This project is based on Walmart sales data. The main goal of this project is to predict weekly sales using Machine Learning.

In this project, I worked with historical sales data and tried different Machine Learning models to find which model gives the best prediction results.

## Dataset

I used the Walmart Sales Forecasting dataset.

The dataset contains:

- train.csv
- test.csv
- features.csv
- stores.csv

The datasets were merged using Store, Date, and IsHoliday where required.

## Data Preprocessing

Before training the models, I performed the following preprocessing steps:

- Loaded the required CSV files using Pandas
- Merged the sales, features, and store datasets
- Handled missing values
- Converted the Date column into Year, Month, and Day
- Removed the original Date column
- Applied One-Hot Encoding to categorical columns
- Split the data into training and testing data

## Models Used

I tried three Machine Learning models:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

I compared the models using:

- MAE
- MSE
- R² Score

After comparing the results, XGBoost gave the best performance, so I selected XGBoost as the final model for the application.

## Model Files

The trained models were saved using Joblib.

```text
models/
├── linear_model.pkl
├── best_random_forest.pkl
└── best_xgboost.pkl