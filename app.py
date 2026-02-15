import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Configuration
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("Telco Customer Churn Prediction")

# Preprocessing
def preprocess_input(df):
    df_clean = df.copy()
    
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

# Sidebar with Input
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])

# Main Execution
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV Loaded")
    
    with st.expander("View Raw Input"):
        st.dataframe(df_raw.head())

    try:
        # Separate Target
        y_test = None
        X_input = df_raw
        if 'Churn' in df_raw.columns:
            # Normalize Churn to 0/1 (Handles 'Yes'/'No' or 1/0)
            y_test = df_raw['Churn'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
            X_input = df_raw.drop('Churn', axis=1)

        # Data Preprocessing
        X_test = preprocess_input(X_input)

        # Align Columns
        if os.path.exists('model/model_columns.pkl'):
            model_columns = joblib.load('model/model_columns.pkl')
            # Reindex forces the test data to have EXACTLY the same columns as training
            # Fills missing columns with 0, drops extra columns
            X_test = X_test.reindex(columns=model_columns, fill_value=0)
        else:
            st.error("Model columns file not found. Please run 'model/training_script.py' locally first.")
            st.stop()

        # Scale Data
        if os.path.exists('model/scaler.pkl'):
            scaler = joblib.load('model/scaler.pkl')
            X_test_scaled = scaler.transform(X_test)
        else:
            st.warning("Scaler not found. Using local scaler.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)

        # Model Selection
        st.sidebar.header("2. Model Selection")
        model_map = {
            "Logistic Regression": "model/logistic_regression.pkl",
            "Decision Tree": "model/decision_tree.pkl",
            "KNN": "model/knn.pkl",
            "Naive Bayes": "model/naive_bayes.pkl",
            "Random Forest": "model/random_forest.pkl",
            "XGBoost": "model/xgboost.pkl"
        }
        selected_model = st.sidebar.selectbox("Choose Model:", list(model_map.keys()))
        model_path = model_map[selected_model]

        # 6. Prediction
        if st.sidebar.button("Run Prediction"):
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                try: y_prob = model.predict_proba(X_test_scaled)[:, 1]
                except: y_prob = y_pred

                st.divider()
                st.subheader(f"Results: {selected_model}")

                if y_test is not None:
                    # Calculate Metrics
                    acc = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    mcc = matthews_corrcoef(y_test, y_pred)

                    # Display Metrics
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Accuracy", f"{acc:.4f}")
                    m2.metric("AUC", f"{auc:.4f}")
                    m3.metric("Precision", f"{prec:.4f}")
                    m4.metric("Recall", f"{rec:.4f}")
                    m5.metric("F1 Score", f"{f1:.4f}")
                    m6.metric("MCC", f"{mcc:.4f}")

                    # Visuals
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Confusion Matrix**")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                    with c2:
                        st.write("**Classification Report**")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
                else:
                    st.success("Predictions generated! (No ground truth found for evaluation)")
                    st.dataframe(pd.DataFrame(y_pred, columns=["Predicted Churn"]))
            else:
                st.error(f"Model file not found at {model_path}. Please run training script.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file in the sidebar to begin.")
