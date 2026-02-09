import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page config for a premium feel
st.set_page_config(
    page_title="Customer Segmentation Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
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
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f1f1f;
        font-family: 'Inter', sans-serif;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_resources():
    model_path = '/Applications/Clusterss/kmeans_model_v2.pkl'
    scaler_path = '/Applications/Clusterss/scaler_v2.pxl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_resources()

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('/Applications/Clusterss/Mall_Customers copy.csv')
    return data

df = load_data()

# Sidebar for Input
st.sidebar.header("User Input Features")
st.sidebar.markdown("Enter details to predict customer segment")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.sidebar.number_input("Annual Income (k$)", min_value=1, max_value=200, value=50)
    spending_score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    data = {
        'Age': age,
        'Annual Income (k$)': income,
        'Spending Score (1-100)': spending_score
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Header
st.title("🛍️ Mall Customer Segmentation")
st.markdown("Unlock deep insights into your customer base using AI-powered clustering.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Prediction", "🔍 Cluster Analysis"])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Avg Age", round(df['Age'].mean(), 1))
    col3.metric("Avg Income", f"${round(df['Annual Income (k$)'].mean(), 1)}k")
    col4.metric("Avg Spending Score", round(df['Spending Score (1-100)'].mean(), 1))
    
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.write(df.describe().T)
    
    st.subheader("Averge Metrics by Gender")
    gender_stats = df.groupby('Genre')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    st.table(gender_stats)

with tab2:
    st.subheader("Predict Customer Segment")
    
    if model and scaler:
        # Scale the input
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        # Mapping clusters to labels
        cluster_labels = {
            0: "Standard (Average Income, Average Spending)",
            1: "Target/Elite (High Income, High Spending)",
            2: "Improviser (Low Income, High Spending)",
            3: "Careful/Conservative (High Income, Low Spending)",
            4: "Sensible (Low Income, Low Spending)"
        }
        
        cluster_colors = {
            0: "#7f8c8d", # Gray
            1: "#27ae60", # Green
            2: "#e74c3c", # Red
            3: "#f39c12", # Orange
            4: "#2980b9"  # Blue
        }
        
        st.markdown(f"""
            <div style="padding: 2.5rem; border-radius: 15px; background-color: {cluster_colors[prediction]}; color: white; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2);">
                <h1 style='color: white; margin-bottom: 0;'>Category: {prediction}</h1>
                <h3 style='color: white; font-weight: 300;'>{cluster_labels[prediction]}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Insight:** Based on the features provided (Age: {input_df['Age'][0]}, Income: {input_df['Annual Income (k$)'][0]}, Score: {input_df['Spending Score (1-100)'][0]}), this customer falls into Segment {prediction}.")
        
    else:
        st.error("Model or Scaler not found! Please ensure your training files are processed.")

with tab3:
    st.subheader("Deeper Dive into Segments")
    
    if model and scaler:
        # Re-calculating clusters for the whole dataset
        df_scaled = scaler.transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        df['Cluster'] = model.predict(df_scaled)
        
        # Grouped analysis
        cluster_stats = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
        st.table(cluster_stats)
        
        # Detailed descriptions
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.info("**Cluster 1: Elite**\n\nHigh earners who spend a lot. They are the prime targets for luxury promotions and exclusive offers.")
            st.warning("**Cluster 2: Improvisers**\n\nLow income but high spending. Often impulsive buyers or younger demographics. Potential for credit products.")
            st.success("**Cluster 0: Standard**\n\nBalanced earners and spenders. The steady majority of the customer base.")
            
        with col_m2:
            st.error("**Cluster 3: Careful**\n\nHigh earners but very conservative spenders. They might be waiting for the right discount or specific needs.")
            st.markdown("""<div style="padding: 15px; border-radius: 5px; background-color: #e9ecef; border-left: 5px solid #6c757d;">
                <strong>Cluster 4: Sensible</strong><br>
                Economical in both earning and spending. Value-oriented customers.
            </div>""", unsafe_allow_html=True)
            
    else:
        st.info("Run clustering model to see analysis.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed with ❤️ for Mall Analytics | © 2024</p>", unsafe_allow_html=True)
