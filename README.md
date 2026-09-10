# Customer Churn Prediction Using Machine Learning

Predicts whether a telecom customer is likely to churn (cancel their subscription) using historical customer data. The project covers the full pipeline: data cleaning, exploratory data analysis, handling class imbalance, model training/tuning, and a saved model ready for inference.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset — 7,043 customers, 21 features including demographics, account information (tenure, contract type, payment method), subscribed services (internet, phone, streaming, tech support), billing charges, and the target label `Churn`.

## Approach

1. **Data cleaning**
   - Dropped the non-predictive `customerID` column
   - Fixed `TotalCharges` (stored as text with blank entries) by converting to numeric and imputing missing values as `0.0`
   - Checked for and confirmed no other missing values

2. **Exploratory Data Analysis**
   - Distribution plots and boxplots for numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`)
   - Value counts for categorical features
   - Correlation heatmap for numerical features
   - Identified significant class imbalance in the target (`Churn`)

3. **Preprocessing**
   - Label-encoded the target (`Yes`/`No` → `1`/`0`)
   - Label-encoded all categorical feature columns, saving the fitted encoders (`encoders.pkl`) for reuse at inference time
   - Train/test split (80/20)
   - Applied **SMOTE** (Synthetic Minority Oversampling Technique) to the training set to correct class imbalance

4. **Modeling**
   - Compared three tree-based classifiers with 5-fold cross-validation on the SMOTE-balanced training data:
     - Decision Tree
     - Random Forest
     - XGBoost
   - Hyperparameter tuning for all three models via `RandomizedSearchCV`
   - Selected the best-performing model based on cross-validation accuracy

5. **Evaluation & Inference**
   - Evaluated the best model on the held-out test set (accuracy, confusion matrix, classification report)
   - Saved the trained model and feature names (`customer_churn_model.pkl`) with `pickle`
   - Included a demo predictive system that loads the saved model/encoders and predicts churn for a new customer record

## Results

| Model | Cross-Validation Accuracy |
|---|---|
| **Random Forest (best)** | **0.84** |
| Decision Tree | — |
| XGBoost | — |

Best model: `RandomForestClassifier(bootstrap=False, min_samples_leaf=2, random_state=42)`, achieving ~0.78 accuracy on the held-out test set.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Decision Tree, Random Forest, model selection & metrics)
- XGBoost
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn (EDA visualizations)

## Project Structure

```
├── Customer_Churn_Prediction_Using_Machine_Learning.ipynb   # Full notebook: EDA, training, evaluation
├── encoders.pkl              # Saved label encoders for categorical features
├── customer_churn_model.pkl  # Saved best model + feature names
└── README.md
```

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Run the notebook
1. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project directory (update the path in the notebook if needed).
2. Open and run `Customer_Churn_Prediction_Using_Machine_Learning.ipynb` cell by cell (originally built in Google Colab).
3. This will regenerate `encoders.pkl` and `customer_churn_model.pkl`.

### Make a prediction with the saved model
```python
import pickle
import pandas as pd

# load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# example customer
input_data = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}
df = pd.DataFrame([input_data])

for col, encoder in encoders.items():
    df[col] = encoder.transform(df[col])

prediction = model.predict(df)
print("Churn" if prediction[0] == 1 else "No Churn")
```

## Future Improvements
- Try additional models (LightGBM, CatBoost, stacking/ensembling)
- Add SHAP/feature-importance analysis for interpretability
- Wrap the model in a simple API or Streamlit app for interactive predictions
- Track experiments with MLflow

## License
MIT (or update to match your preference)
