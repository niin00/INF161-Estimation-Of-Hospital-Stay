### How to run the code
- 
---
# Hospital Length of Stay Prediction

## Project Overview
This project aims to predict the expected length of hospital stay for patients based on demographic, physiological, and disease severity data. The goal is to build a robust machine learning model that can aid hospitals in resource planning by forecasting patient stay durations.

## Project Structure
The project is organized into the following steps:
1. **Data Preparation**
2. **Feature Engineering**
3. **Exploratory Data Analysis (EDA)**
4. **Modeling and Evaluation**
5. **Hyperparameter Tuning and Final Model Selection**
6. **Deployment**

## Dataset Description
The project uses four datasets containing patient information:

1. `demographics.csv`: Demographic details like age, gender, education, income, and ethnicity.
2. `hospital.csv`: Contains the length of hospital stay (`oppholdslengde`) and whether the patient died in the hospital.
3. `physiological.txt`: Physiological measurements such as heart rate, blood pressure, and respiratory rate.
4. `severity.json`: Nested data on comorbidities, disease severity, and survival estimates.

## Data Preparation

1. **Data Cleaning**:
   - Placeholder values (e.g., `-99`) in the `hospital.csv` dataset were replaced with `NaN` and imputed to avoid skewing analysis.
   - Missing values in `demographics.csv` were imputed using the median for continuous variables and the mode for categorical variables.
   - Physiological data from `physiological.txt` was imputed using the median for numerical columns.

2. **Merging Datasets**:
   - All datasets were merged on the `pasient_id` key to create a single dataset containing demographic, physiological, hospital, and severity data.
   - For `severity.json`, relevant sections (e.g., `sykdomskategori.0` for disease categories) were flattened and merged.

3. **Imputation of Remaining Missing Values**:
   - Any remaining missing values in the merged dataset were imputed with the median for numerical columns and the mode for categorical columns.

4. **Data Splitting**:
   - The consolidated dataset was split into training (52.5%), validation (17.5%), and test sets (30%).

## Feature Engineering

1. **New Feature Creation**:
   - **Age Grouping**: The `alder` (age) column was grouped into categories (e.g., `0-18`, `19-40`, etc.) to better capture age-based trends.
   - **Comorbidity Count**: A new feature, `komorbiditet_antall`, was created to represent the number of comorbidities for each patient.

2. **Encoding Categorical Variables**:
   - One-hot encoding was applied to categorical variables such as gender (`kjønn`), ethnicity (`etnisitet`), age group (`alder_gruppe`), income (`inntekt`), and disease category (`sykdomskategori.0`).

3. **Feature Scaling**:
   - Numerical columns (e.g., `alder`, `utdanning`, `blodtrykk`, etc.) were standardized using `StandardScaler` to ensure consistency across features.

4. **Reducing Redundancy**:
   - Highly correlated features were identified with a threshold of 0.85, and one feature from each highly correlated pair was removed to minimize multicollinearity.

## Exploratory Data Analysis (EDA)
- **Correlation Analysis**: A heatmap was used to visualize relationships between numerical features and identify multicollinearity.
- **Distribution of Length of Stay**: The target variable (`oppholdslengde`) exhibited a right-skewed distribution, with most patients having shorter stays and a few experiencing significantly longer stays.
- **Boxplots and Scatter Plots**: Plots such as age vs. length of stay and gender vs. length of stay were analyzed to understand potential relationships and identify patterns in hospital stay lengths.

## Modeling and Evaluation

1. **Initial Model Evaluation**:
   - Multiple models were tested, including:
     - **Linear Regression**
     - **Random Forest**
     - **Support Vector Regression (SVR)**
     - **Gradient Boosting**
     - **XGBoost**
   - Each model was evaluated on the validation and test sets using metrics like RMSE, MAE, and R-squared. Additionally, cross-validation was performed for each model to ensure consistency.

2. **Model Performance Summary**:
   - **Gradient Boosting** outperformed the other models with the lowest RMSE and MAE on the test set.
   - Cross-validation results indicated that Gradient Boosting achieved consistent performance across different folds, supporting its selection as the final model.

## Hyperparameter Tuning and Final Model Selection

1. **Hyperparameter Tuning**:
   - A GridSearchCV was used to tune the hyperparameters for Gradient Boosting. The best parameters were:
     - `learning_rate`: 0.01
     - `max_depth`: 3
     - `min_samples_leaf`: 4
     - `min_samples_split`: 2
     - `n_estimators`: 200
     - `subsample`: 0.8

2. **Final Model Evaluation**:
   - **Validation Performance**:
     - RMSE: 22.82
     - MAE: 12.11
     - R-squared: 0.085
   - **Test Performance**:
     - RMSE: 21.26
     - MAE: 11.97
     - R-squared: 0.107

3. **Feature Importance Analysis**:
   - The final model's feature importances were plotted to understand the relative contribution of each feature to the model’s predictions.

## Deployment

1. **Saving the Model**:
   - The final tuned Gradient Boosting model was saved using `joblib` as `gradient_boosting_model.pkl` for easy loading and use in deployment.

2. **Usage Instructions**:
   - To use the model, load it using `joblib.load('gradient_boosting_model.pkl')`.
   - Prepare input data in the same format as the training data, including feature scaling and encoding as required.
   - Use the model’s `predict()` method on new data to get predictions for hospital stay length.

### Example Usage

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('gradient_boosting_model.pkl')

# Prepare sample data (ensure proper encoding and scaling)
sample_data = pd.DataFrame({
    'age': [65],
    'gender_male': [1],
    'heart_rate': [78],
    # Include other features as required
})

# Predict length of stay
prediction = model.predict(sample_data)
print(f"Predicted Length of Stay: {prediction[0]} days")
```

## Additional Information

1. **Limitations**:
   - This model may have limitations with outliers or specific subpopulations due to data skewness. Adjustments such as further tuning or using ensemble techniques could improve future models.

2. **Future Work**:
   - Incorporating additional data (e.g., more granular disease progression metrics) could potentially improve accuracy. Exploring deep learning models may also enhance performance for complex relationships.

3. **Contact Information**:
   - For questions or contributions, please contact [Your Contact Information].

---

This README should cover all relevant details. Let me know if you would like further customization or have any additional points to include!