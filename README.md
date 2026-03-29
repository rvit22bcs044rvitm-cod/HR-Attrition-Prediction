# HR Attrition Prediction (Machine Learning)

## Project Overview
This project uses the IBM HR Analytics dataset to predict employee attrition using classification algorithms. 
The goal is to help HR identify 'at-risk' employees and understand the key drivers of turnover.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, Seaborn, Imbalanced-Learn (SMOTE)
* **Environment:** Google Colab

## Key Steps Implemented
1. **Data Cleaning:** Removed zero-variance columns (StandardHours, Over18).
2. **EDA:** Identified Overtime and Salary as major predictors.
3. **Handling Imbalance:** Applied SMOTE to training data only.
4. **Modeling:** Logistic Regression, Random Forest, KNN, Decision Tree.
5. **Evaluation:** Achieved 85.7% Accuracy and 0.68 ROC-AUC.

## Final Insights
The top drivers for attrition were **Stock Option Level**, **Monthly Income**, and **Job Satisfaction**.
"""

