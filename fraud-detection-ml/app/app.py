import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Page title
st.title("💳 Credit Card Fraud Detection Dashboard")

st.write("Upload a dataset to detect fraudulent transactions.")

# Load trained model
model = joblib.load("model/fraud_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Remove label column if present
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()

    # Apply scaling (same preprocessing used in training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predictions
    predictions = model.predict(X_scaled)

    df["Fraud_Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(df.head())

    fraud_count = df["Fraud_Prediction"].sum()

    st.subheader("Fraud Detection Summary")

    st.write(f"Total Transactions: {len(df)}")
    st.write(f"Fraud Transactions Detected: {fraud_count}")
    st.write(f"Fraud Rate: {fraud_count / len(df):.4f}")

    # Show fraud transactions
    fraud_df = df[df["Fraud_Prediction"] == 1]

    if len(fraud_df) > 0:
        st.subheader("Detected Fraud Transactions")
        st.dataframe(fraud_df)
    else:
        st.success("No fraud detected in uploaded data.")