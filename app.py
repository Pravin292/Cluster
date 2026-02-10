import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    h1 {
        color: #1f1f1f;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_resources():
    model_path = "kmeans_model_v2.pkl"
    scaler_path = "scaler_v2.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_resources()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers copy.csv")

df = load_data()

# Sidebar input
st.sidebar.header("User Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.number_input("Age", 18, 100, 30)
    income = st.sidebar.number_input("Annual Income (k$)", 1, 200, 50)
    spending_score = st.sidebar.number_input("Spending Score (1-100)", 1, 100, 50)

    data = {
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending_score,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main title
st.title("🛍️ Mall Customer Segmentation")

tab1, tab2, tab3 = st.tabs(["Overview", "Prediction", "Cluster Analysis"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Avg Age", round(df["Age"].mean(), 1))
    col3.metric("Avg Income", round(df["Annual Income (k$)"].mean(), 1))
    col4.metric("Avg Spending", round(df["Spending Score (1-100)"].mean(), 1))

    st.dataframe(df.head())

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Predict Customer Segment")

    if model and scaler:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"Predicted Cluster: {prediction}")
    else:
        st.error("Model or Scaler not found!")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Cluster Analysis")

    if model and scaler:
        df_scaled = scaler.transform(
            df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        )
        df["Cluster"] = model.predict(df_scaled)

        cluster_stats = (
            df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
            .mean()
            .reset_index()
        )

        st.table(cluster_stats)
    else:
        st.info("Model not loaded.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
