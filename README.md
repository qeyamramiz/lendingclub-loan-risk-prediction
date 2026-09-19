# LendingClub Loan Risk Prediction

## Project Overview

This project uses machine learning to predict whether a LendingClub loan
will be fully paid or become a risky loan.

The project includes data cleaning, exploratory data analysis (EDA),
feature preprocessing, model training, model evaluation, threshold tuning,
and a prediction pipeline for new loan applications.

## Dataset

The dataset used in this project is the LendingClub loan dataset from Kaggle.

The original dataset contains approximately 2.26 million loan records and
151 features.

For this project, the data was cleaned and processed by handling missing
values, removing redundant and leakage-prone features, transforming date
features, encoding categorical variables, and preparing the data for
machine learning.

The raw dataset is not included in this repository because of its large size.

## Data Cleaning & Preprocessing

The following steps were performed to prepare the data for machine learning:

- Removed features with excessive missing values.
- Handled remaining missing values using appropriate strategies.
- Removed redundant and highly correlated features.
- Removed features that could cause data leakage.
- Converted date features into useful numerical features.
- Encoded categorical variables using one-hot encoding.
- Created a binary target variable:
  - `0` = Fully Paid
  - `1` = Charged Off / Default
- Split the data into training and testing sets using an 80/20 split.
- Used stratified sampling to preserve the target-class distribution.
- Standardized continuous numerical features using `StandardScaler`.

## Model Training & Evaluation

Three classification models were trained and evaluated:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

The Gradient Boosting model was further evaluated using different
classification thresholds. A threshold of `0.55` was selected based on
the highest F1-score among the tested thresholds.

### Final Model Results

| Metric | Value |
|---|---:|
| Accuracy | 0.71 |
| Class 1 Precision | 0.36 |
| Class 1 Recall | 0.60 |
| Class 1 F1-score | 0.45 |
| ROC-AUC | 0.738 |
| PR-AUC | 0.420 |
| Classification Threshold | 0.55 |

## Project Structure

```text
LendingClub project/
│
├── data/
│   └── README.md
│
├── models/
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   ├── threshold.pkl
│   ├── feature_columns.pkl
│   └── continuous_features.pkl
│
├── 01_data_cleaning.ipynb
├── 02_eda.ipynb
├── 03_ml_preprocessing.ipynb
├── 04_model_testing.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd LendingClub-project

### 2. Create a virtual environment

```bash
python -m venv .venv

### 3. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Run the notebooks

Open the notebooks in Jupyter Notebook or VS code:
01_data_cleaning.ipynb
02_eda.ipynb
03_ml_preprocessing.ipynb
04_model_testing.ipynb

## Prediction Pipeline

The project includes a prediction function that can evaluate a single loan application.

The prediction pipeline:

1. Receives loan information.
2. Converts the information into a DataFrame.
3. Ensures the features have the correct order.
4. Scales the continuous features using the saved scaler.
5. Generates a risk probability using the trained model.
6. Applies the saved classification threshold (`0.55`).
7. Returns the prediction and risk probability.

Example output:

```text
Risk probability: 20.15%
Prediction: 0
Result: Low-risk Loan


## Saved Model Files

The `models/` directory contains the artifacts required for prediction:

- `gradient_boosting_model.pkl` — trained Gradient Boosting model
- `scaler.pkl` — fitted feature scaler
- `threshold.pkl` — classification threshold
- `feature_columns.pkl` — expected feature names and order
- `continuous_features.pkl` — features that require scaling

## Future Improvements

- Build a web application for interactive loan-risk predictions.
- Add additional model tuning and cross-validation.
- Improve probability calibration.
- Add model explainability using feature importance or SHAP.
- Deploy the prediction application.