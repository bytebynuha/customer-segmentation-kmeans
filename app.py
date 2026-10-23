
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("Customer Segmentation Using K-Means")
st.write(
    "Group customers based on age, annual income, "
    "and spending score."
)

# Create folders for our data and saved predictions
os.makedirs("data", exist_ok=True)
os.makedirs("storage", exist_ok=True)

DATA_FILE = "data/customers.csv"
HISTORY_FILE = "storage/predictions.csv"

# Generate demo data if the dataset does not exist
if not os.path.exists(DATA_FILE):
    rng = np.random.default_rng(42)

    groups = [
        (30, 30, 25),
        (35, 75, 80),
        (45, 65, 30),
        (25, 35, 75),
    ]

    records = []

    for age_mean, income_mean, score_mean in groups:
        for _ in range(50):
            records.append({
                "Age": int(np.clip(
                    rng.normal(age_mean, 6), 18, 75
                )),
                "Annual Income (k$)": round(
                    float(np.clip(
                        rng.normal(income_mean, 8), 10, 150
                    )), 2
                ),
                "Spending Score (1-100)": int(np.clip(
                    rng.normal(score_mean, 10), 1, 100
                ))
            })

    pd.DataFrame(records).to_csv(DATA_FILE, index=False)

# Load customer dataset
df = pd.read_csv(DATA_FILE)

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

st.subheader("1. Customer Dataset")
st.write(f"Number of customers: {len(df)}")
st.dataframe(df.head(10), use_container_width=True)

# Prepare and scale features
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
st.subheader("2. Choose K using the Elbow Method")

inertias = []

for k_value in range(1, 11):
    kmeans = KMeans(
        n_clusters=k_value,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), inertias, marker="o")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)
plt.close(fig1)

st.caption(
    "Look for the bend in the curve where the reduction "
    "in inertia starts slowing down."
)

# Select K
k = st.slider(
    "Select the number of clusters",
    min_value=2,
    max_value=10,
    value=4
)

# Train the selected K-Means model
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)

# Visualize customer clusters
st.subheader("3. Visualize Customer Clusters")

fig2, ax2 = plt.subplots()

scatter = ax2.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.75
)

ax2.set_xlabel("Annual Income (thousands of dollars)")
ax2.set_ylabel("Spending Score")
ax2.set_title("Customer Clusters")

fig2.colorbar(scatter, ax=ax2, label="Cluster")
st.pyplot(fig2)
plt.close(fig2)

# Cluster summary
st.subheader("4. Cluster Summary")

summary = df.groupby("Cluster")[FEATURES].mean().round(2)
st.dataframe(summary, use_container_width=True)

# Predict the cluster of a new customer
st.subheader("5. Predict a New Customer's Cluster")

with st.form("new_customer"):
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=25
    )

    income = st.number_input(
        "Annual Income (thousands of dollars)",
        min_value=0.0, max_value=500.0, value=50.0
    )

    score = st.slider(
        "Spending Score", min_value=1, max_value=100, value=60
    )

    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    new_customer_data = pd.DataFrame([{
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": score
    }])

    scaled_customer = scaler.transform(new_customer_data[FEATURES])
    predicted_cluster = int(model.predict(scaled_customer)[0])

    st.success(
        f"The new customer belongs to Cluster {predicted_cluster}."
    )

    new_customer_data["Predicted Cluster"] = predicted_cluster

    if os.path.exists(HISTORY_FILE):
        new_customer_data.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_customer_data.to_csv(HISTORY_FILE, index=False)

# Show prediction history
st.subheader("6. Prediction History")

if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
    st.dataframe(history, use_container_width=True)
else:
    st.info("No predictions saved yet. Try the prediction form above.")
