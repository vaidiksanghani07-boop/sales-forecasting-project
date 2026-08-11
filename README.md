# Walmart Sales Forecasting using Machine Learning

## Project Overview

This project is a Machine Learning based Sales Forecasting system developed using the Walmart Store Sales dataset. The main objective is to predict `Weekly_Sales` using historical sales data and related store, holiday, economic, and markdown features.

The project follows a complete Machine Learning workflow:

**Data Collection → Data Preprocessing → EDA → Feature Engineering → Model Training → Hyperparameter Tuning → Model Comparison → Best Model Selection → Model Saving → Streamlit Application**

## 📊Dataset

The project uses the Walmart Store Sales dataset containing:

- Store and Department information
- Weekly Sales
- Date and Holiday information
- Temperature
- Fuel Price
- CPI
- Unemployment
- Markdown features
- Store Type and Size

The `train.csv`, `features.csv`, and `stores.csv` files were merged and processed before model training.

## Data Preprocessing & EDA

The following steps were performed:

- Merged the required datasets
- Handled missing values
- Converted Date into datetime format
- Extracted Year, Month, and Day
- Removed the original Date column
- Applied One-Hot Encoding to categorical variables
- Created features (`X`) and target (`y`)
- Performed Train-Test Split
- Performed Exploratory Data Analysis
- Analyzed correlations and sales patterns

## Models Used

Three regression models were trained and compared:

1. **Linear Regression** – Used as the baseline model
2. **Random Forest Regressor** – Used to capture nonlinear relationships
3. **XGBoost Regressor** – Tuned and selected as the final model

Hyperparameter tuning was performed for Random Forest and XGBoost using `RandomizedSearchCV`.

## 📈 Model Results

 Linear Regression -> MSE -460,782,215.62 | MAE - 14,377.53 | |R2 socre - 0.1164 |
Random Forest  -> MSE - 225,985,700.97 | MAE - 9,253.41 |  |R2 score - 0.5666 |
**XGBoost**  -> **MSE - 58,083,207.55 |MAE - 4,590.57|  | R2 score - 0.8886 |**
**


###  Best Model : XGBoost

XGBoost achieved the best overall performance with:

- MAE:4,590.57
- **R² Score**: 0.8886

The model explains approximately **88.86% of the variation in Weekly Sales** on the test dataset. Therefore, XGBoost was selected as the final model.

##  Model Saving

The final trained model was saved as a `.pkl` file using Joblib.

```python
import joblib

joblib.dump(best_xgb, "best_xgboost.pkl")