# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts customer churn using the Telco Customer Churn dataset. The notebook covers data cleaning, exploratory data analysis, handling class imbalance, training and tuning multiple classifiers, and building a predictive system with the saved model.

## Dataset

- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Shape: 7,043 rows x 21 columns
- Target column: `Churn` (`Yes` / `No`)

## Workflow

**1. Data Cleaning**
- Dropped `customerID` (not useful for modeling)
- Converted `TotalCharges` from text to numeric, replacing blank entries with `0.0`
- Checked for missing values

**2. Exploratory Data Analysis**
- Distribution plots and boxplots for `tenure`, `MonthlyCharges`, `TotalCharges`
- Value counts for categorical columns
- Correlation heatmap for numerical features
- Checked class distribution of `Churn` -> found the dataset imbalanced

**3. Preprocessing**
- Encoded `Churn` as `1`/`0`
- Label-encoded categorical feature columns and saved the encoders to `encoders.pkl`
- Split features (`X`) and target (`y`)
- Train/test split: 80/20 (`random_state=42`)

**4. Handling Class Imbalance**
- Applied SMOTE to the training set to balance the classes

**5. Model Training**
- Trained and compared three models with 5-fold cross-validation on the SMOTE-balanced training data:
  - Decision Tree
  - Random Forest
  - XGBoost

**6. Hyperparameter Tuning**
- Tuned all three models using `RandomizedSearchCV`
- Selected the model with the best cross-validation score

**7. Evaluation**
- Evaluated the best model on the test set using accuracy, confusion matrix, and classification report

**8. Saving the Model**
- Saved the best model to `best_model.pkl`
- Saved the model along with feature names to `customer_churn_model.pkl`

**9. Predictive System**
- Loaded `customer_churn_model.pkl` and `encoders.pkl`
- Built a single-customer input example, encoded it, and predicted churn with probability

## Results

- Best model: `RandomForestClassifier(bootstrap=False, min_samples_leaf=2, random_state=42)`
- Best cross-validation accuracy: **0.84**
- Test set accuracy: **0.78**

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## Files

- `Customer_Churn_Prediction_Using_Machine_Learning.ipynb` - full notebook
- `encoders.pkl` - saved label encoders for categorical columns
- `customer_churn_model.pkl` - saved model + feature names
- `best_model.pkl` - saved best model
