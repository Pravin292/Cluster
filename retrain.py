import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('/Applications/Clusterss/Mall_Customers copy.csv')

# Features now include Age
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Retraining with 5 clusters (as selected previously)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save the new resources
joblib.dump(scaler, '/Applications/Clusterss/scaler_v2.pxl')
joblib.dump(kmeans, '/Applications/Clusterss/kmeans_model_v2.pkl')

print("Retraining completed with Age, Income, and Spending Score.")
