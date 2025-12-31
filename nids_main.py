import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("## 🛡️ AI-Based Network Intrusion Detection System")
st.markdown(
    "Machine Learning–driven intrusion detection using the **CIC-IDS-2017 benchmark dataset**"
)
st.divider()

# --------------------------------------------------
# LOAD & PREPROCESS DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    # Fix column spacing issue
    df.columns = df.columns.str.strip()

    # Binary encoding
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Keep numeric features
    df = df.select_dtypes(include=[np.number])

    # Handle NaN & Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Control Panel")
st.sidebar.write("Train and evaluate the intrusion detection model.")
train_button = st.sidebar.button("🚀 Train Model")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
with st.spinner("Loading CIC-IDS-2017 dataset..."):
    df = load_data()

st.success("Dataset loaded successfully")

# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{df.shape[0]:,}")
c2.metric("Total Features", df.shape[1] - 1)
c3.metric("Attack Samples", int(df["Label"].sum()))

st.divider()

# --------------------------------------------------
# SPLIT FEATURES & LABEL
# --------------------------------------------------
X = df.drop("Label", axis=1)
y = df["Label"]

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
if train_button:
    with st.spinner("Training Random Forest model..."):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)

        st.session_state.update({
            "model": model,
            "X": X,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob
        })

    st.success("Model trained successfully")

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------
    st.subheader("📊 Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Error Rate", f"{1 - acc:.4f}")

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------
    st.subheader("🔎 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Intrusion"],
        yticklabels=["Normal", "Intrusion"],
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=["Normal", "Intrusion"]))

    # --------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------
    st.subheader("📈 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # --------------------------------------------------
    # PRECISION–RECALL CURVE
    # --------------------------------------------------
    st.subheader("📉 Precision–Recall Curve")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall Curve")
    st.pyplot(fig_pr)

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------
    st.subheader("🔍 Top Feature Importance")

    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=feat_df,
        x="Importance",
        y="Feature",
        ax=ax_fi
    )
    ax_fi.set_title("Top 10 Influential Features")
    st.pyplot(fig_fi)

# --------------------------------------------------
# LIVE TRAFFIC SIMULATION
# --------------------------------------------------
st.divider()
st.subheader("🔴 Live Traffic Simulation")

if "model" not in st.session_state:
    st.info("Train the model to enable live traffic simulation.")
else:
    if st.button("Simulate Incoming Traffic"):
        sample = X.sample(1, random_state=np.random.randint(0, 100000))
        prediction = st.session_state["model"].predict(sample)[0]

        st.write("📥 Incoming Network Flow (Sampled)")
        st.dataframe(sample)

        if prediction == 1:
            st.error("🚨 Intrusion Detected")
        else:
            st.success("✅ Normal Traffic Detected")

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------
st.divider()
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "AI-Based Network Intrusion Detection System | CIC-IDS-2017 | Random Forest Classifier"
)
