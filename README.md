# Telco Customer Churn Classification

## a. Problem Statement

The telecommunications industry faces a significant challenge in customer retention. Acquiring new customers is often more expensive than retaining existing ones. The goal of this project is to build a machine learning classification pipeline to predict customer churn (whether a customer will leave the service) based on their demographic, account, and service usage data.

This is a **binary classification problem** where the target variable is **Churn (Yes/No)**. By identifying at-risk customers early, the business can take proactive measures to retain them.

---

## b. Dataset Description

**Dataset Name:** Telco Customer Churn  
**Source:** Kaggle / IBM Sample Data Sets  

### Dataset Structure

- **Instances:** 7,043 rows  
- **Features:** 21 columns  
- **Target Variable:** Churn  

### Key Features

**Demographics**
- Gender  
- SeniorCitizen  
- Partner  
- Dependents  

**Services**
- PhoneService  
- MultipleLines  
- InternetService  
- OnlineSecurity  
- OnlineBackup  
- DeviceProtection  
- TechSupport  
- StreamingTV  
- StreamingMovies  

**Account Information**
- Contract  
- PaperlessBilling  
- PaymentMethod  
- MonthlyCharges  
- TotalCharges  
- Tenure  

### Preprocessing Steps

- Categorical variables were transformed using **One-Hot Encoding** to handle nominal data robustly.  
- Numerical variables were scaled using **StandardScaler**.  
- The dataset was split into **Training (80%)** and **Testing (20%)** sets.  

---

## c. Models Used & Comparison

We implemented **six machine learning models** to solve this classification problem. Below is the performance comparison.

### Model Performance Metrics

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|---------------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression      | 0.8204   | 0.8620 | 0.6852    | 0.5952 | 0.6370   | 0.5208 |
| Decision Tree            | 0.8055   | 0.8412 | 0.6645    | 0.5362 | 0.5935   | 0.4722 |
| KNN                      | 0.7615   | 0.7812 | 0.5536    | 0.5121 | 0.5320   | 0.3729 |
| Naive Bayes              | 0.6962   | 0.8376 | 0.4607    | 0.8633 | 0.6007   | 0.4406 |
| Random Forest (Ensemble) | 0.8070   | 0.8579 | 0.6784    | 0.5147 | 0.5854   | 0.4701 |
| XGBoost (Ensemble)       | 0.8027   | 0.8557 | 0.6578    | 0.5308 | 0.5875   | 0.4644 |

---

## d. Observations on Model Performance

| ML Model Name             | Observation |
|---------------------------|-------------|
| Logistic Regression      | Performed exceptionally well with the highest AUC (0.8620) and highest Accuracy (0.8204). It offers a very strong baseline with balanced metrics across the board. |
| Decision Tree            | Competitive accuracy (0.8055) with solid precision. Hyperparameter tuning (pruning) helped maintain generalization compared to ensemble methods. |
| KNN                      | The weakest performer (Accuracy ≈ 0.76, MCC ≈ 0.37). It struggled with high dimensionality introduced by One-Hot Encoding, leading to less reliable distance calculations. |
| Naive Bayes              | Achieved the highest Recall (0.8633). It is excellent at catching churners (minimizing false negatives) but suffers from low Precision (0.46), resulting in many false positives. |
| Random Forest (Ensemble) | Very stable and robust with high Accuracy (0.8070) and strong Precision (0.6784). It effectively reduced the variance of individual trees. |
| XGBoost (Ensemble)       | Strong Accuracy (0.8027) and high AUC (0.8557). It provides accurate predictions overall and balances Precision and Recall effectively. |
