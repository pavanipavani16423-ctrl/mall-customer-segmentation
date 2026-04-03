import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🛍 Mall Customer Segmentation using K-Means")
st.markdown("Interactive Machine Learning App for Business Insights")

# Sidebar Controls
st.sidebar.header("⚙ Model Controls")
k = st.sidebar.slider("Select Number of Clusters", 2, 6, 5)

# Load Dataset
df = pd.read_csv("Mall_Customers_Synthetic.csv")

# Feature Selection
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Model
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X)
centroids = model.cluster_centers_

# -----------------------------------
# Dataset Preview Section (Full Width)
# -----------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("📈 Summary Metrics")
colA, colB = st.columns(2)
colA.metric("Total Customers", len(df))
colB.metric("Clusters Selected", k)

st.markdown("---")

# -----------------------------------
# Cluster Visualization (Full Width)
# -----------------------------------
st.subheader("📈 Cluster Visualization")

fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.tab10.colors

for i in range(k):
    cluster_data = df[df["Cluster"] == i]
    ax.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {i}",
        color=colors[i % len(colors)],
        alpha=0.7
    )

# Centroids
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=250,
    color='black',
    label='Centroids'
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()

st.pyplot(fig)

st.markdown("---")

# -----------------------------------
# Cluster Interpretation Section
# -----------------------------------
st.subheader("📌 Cluster Insights")

for i in range(k):
    cluster_group = df[df["Cluster"] == i]
    avg_income = cluster_group["Annual Income (k$)"].mean()
    avg_score = cluster_group["Spending Score (1-100)"].mean()

    st.markdown(f"### Cluster {i}")
    st.write(f"Average Income: {avg_income:.2f}")
    st.write(f"Average Spending Score: {avg_score:.2f}")

    if avg_income < 40 and avg_score < 40:
        st.info("🟡 Budget Customers")
    elif avg_income < 40 and avg_score >= 60:
        st.info("🟠 Impulse Buyers")
    elif 40 <= avg_income <= 70 and avg_score >= 60:
        st.info("🟢 Regular Customers")
    elif avg_income > 70 and avg_score < 40:
        st.info("🔵 Careful Customers")
    elif avg_income > 70 and avg_score >= 60:
        st.success("⭐ Premium Customers")
    else:
        st.info("Moderate Customers")

st.markdown("---")

# -----------------------------------
# Prediction Section
# -----------------------------------
st.subheader("🎯 Predict New Customer Segment")

col3, col4 = st.columns(2)

with col3:
    income = st.number_input("Enter Annual Income (k$)", min_value=0.0)

with col4:
    score = st.number_input("Enter Spending Score (1-100)", min_value=0.0, max_value=100.0)

if st.button("Predict Cluster 🛍️"):

    with st.spinner("🛍️ Analyzing Customer Shopping Behavior..."):
        time.sleep(2)

    new_data = [[income, score]]
    cluster = model.predict(new_data)[0]

    # 🛍️ Shopping Bag Animation
    st.markdown("""
        <style>
        .shopping {
            position: fixed;
            top: -50px;
            font-size: 35px;
            animation: fall 3s linear;
        }

        @keyframes fall {
            0% { transform: translateY(-50px); opacity: 1; }
            100% { transform: translateY(100vh); opacity: 0; }
        }
        </style>

        <div class="shopping" style="left:10%;">🛍️</div>
        <div class="shopping" style="left:30%; animation-delay:0.5s;">🛍️</div>
        <div class="shopping" style="left:50%; animation-delay:1s;">🛍️</div>
        <div class="shopping" style="left:70%; animation-delay:1.5s;">🛍️</div>
        <div class="shopping" style="left:90%; animation-delay:2s;">🛍️</div>
    """, unsafe_allow_html=True)

    st.success(f"🛍️ Assigned to Cluster: {cluster}")