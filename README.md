# 🛍️ Mall Customer Segmentation Pro

An AI-powered web application that segments mall customers into distinct groups based on their demographics and spending patterns. This tool helps businesses understand their customer base to drive targeted marketing strategies.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🚀 Features

- **AI-Powered Clustering**: Uses K-Means algorithm to categorize customers into 5 unique segments.
- **Multidimensional Analysis**: Unlike standard models, this implementation considers **Age**, **Annual Income**, and **Spending Score** as primary dependable features.
- **Real-time Prediction**: Enter specific customer data via the interactive sidebar to instantly predict their cluster.
- **Visual Analytics**: Interactive dashboards showing data distribution and cluster locations.
- **Premium UI**: Modern dark/light mode compatible design with custom CSS for a professional feel.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn (K-Means Clustering)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

## 📁 Project Structure

- `app.py`: The main Streamlit web application.
- `retrain.py`: Utility script used to train the K-Means model using 3 features (Age, Income, Spending Score).
- `kmeans_model_v2.pkl`: The trained clustering model.
- `scaler_v2.pxl`: The fitted StandardScaler for normalizing inputs.
- `Mall_Customers copy.csv`: The dataset used for training and analysis.

## 🚦 Getting Started

### Prerequisites
Make sure you have Python installed. You will need the following libraries:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the App
1. Clone this repository or navigate to the project folder.
2. Run the Streamlit command:
```bash
streamlit run app.py
```

## 🎯 Customer Segments Explained
The model classifies customers into these 5 groups:
1. **Elite (Target)**: High Income, High Spending.
2. **Careful**: High Income, Low Spending.
3. **Standard**: Average Income, Average Spending.
4. **Improvisers**: Low Income, High Spending.
5. **Sensible**: Low Income, Low Spending.

---

Developed with ❤️ for Mall Analytics.
