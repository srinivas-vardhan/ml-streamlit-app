# Imports
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
import warnings

# Configuration
warnings.filterwarnings('ignore')

if os.getcwd().endswith('model'):
    os.chdir('..')
if not os.path.exists('model'): os.makedirs('model')
if not os.path.exists('data'): os.makedirs('data')

# Load Data
def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    data = pd.DataFrame()
    data['tenure'] = np.random.randint(1, 73, n_samples)
    data['MonthlyCharges'] = np.random.uniform(18.25, 118.75, n_samples)
    data['TotalCharges'] = data['tenure'] * data['MonthlyCharges']
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
    data['SeniorCitizen'] = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    data['Partner'] = np.random.choice(['Yes', 'No'], n_samples)
    data['Dependents'] = np.random.choice(['Yes', 'No'], n_samples)
    data['PhoneService'] = np.random.choice(['Yes', 'No'], n_samples)
    data['MultipleLines'] = np.random.choice(['No phone service', 'No', 'Yes'], n_samples)
    data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples)
    data['OnlineSecurity'] = np.random.choice(['No', 'Yes', 'No internet service'], n_samples)
    data['OnlineBackup'] = np.random.choice(['No', 'Yes', 'No internet service'], n_samples)
    data['DeviceProtection'] = np.random.choice(['No', 'Yes', 'No internet service'], n_samples)
    data['TechSupport'] = np.random.choice(['No', 'Yes', 'No internet service'], n_samples)
    data['StreamingTV'] = np.random.choice(['No', 'Yes', 'No internet service'], n_samples)
    data['StreamingMovies'] = np.random.choice(['No', 'Yes', 'No internet service'], n_samples)
    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
    data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], n_samples)
    data['PaymentMethod'] = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples)
    data['Churn'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    return data

def load_data():
    if os.path.exists('data/telco_churn.csv'):
        print("Loading USER SPLIT data (data/telco_churn.csv)...")
        return pd.read_csv('data/telco_churn.csv')
    elif os.path.exists('data/telco_churn_train.csv'):
        print("Loading FULL data (data/telco_churn_train.csv)...")
        return pd.read_csv('data/telco_churn_train.csv')
    else:
        print("No CSV found. Using SYNTHETIC data generator...")
        return generate_synthetic_data()

# Data Preprocessing
def preprocess_training_data(df_input):
    df_clean = df_input.copy()
    
    # Drop ID if exists
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
    
    # Handle Numeric
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].mean(), inplace=True)
    
    # One-Hot Encoding (Robust)
    df_clean = pd.get_dummies(df_clean)
    
    return df_clean

# Main Execution
if __name__ == "__main__":
    
    # Load data
    df = load_data()
    
    # Separate Target
    if 'Churn' in df.columns:
        # Normalize Churn to 0/1 (Handles 'Yes'/'No' or 1/0)
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
        X = df.drop('Churn', axis=1)
    else:
        raise ValueError("Churn column missing!")

    # Data Preprocessing
    X_processed = preprocess_training_data(X)
    print("Preprocessing Complete.")
    print(f"New Feature Count: {X_processed.shape[1]}")

    # Save Columns
    model_columns = list(X_processed.columns)
    joblib.dump(model_columns, 'model/model_columns.pkl')
    print(f"Saved {len(model_columns)} column names to 'model/model_columns.pkl'.")

    # Scaling & Splitting
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Scaler saved to 'model/scaler.pkl'.")

    # Training
    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=10),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.05, n_estimators=200)
    }

    print("Training & Evaluation")
    results_list = []

    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Save Model
        filename = f"model/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        
        # Predict on Test Set
        y_pred = model.predict(X_test_scaled)
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        except:
            y_prob = y_pred 
        
        # Calculate Metrics
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        results_list.append(metrics)
        print(f"{name}: Accuracy = {metrics['Accuracy']:.4f}")

    # Model Comparison
    results_df = pd.DataFrame(results_list)
    print("Model Performance Summary:")
    print(results_df.round(4).to_string(index=False))    
    print("Training Completed. You can now run 'streamlit run app.py'")
