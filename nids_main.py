import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.title("🔐 AI-Based Network Intrusion Detection System")
st.write("Detect malicious network traffic using Machine Learning (Random Forest)")

# --------------------------------------------------
# Load Dataset (YOUR REAL CSV FILES)
# --------------------------------------------------
@st.cache_data
def load_data():
    csv_files = [
        "/Users/simhadrikondapally/Downloads/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "/Users/simhadrikondapally/Downloads/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
        "/Users/simhadrikondapally/Downloads/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "/Users/simhadrikondapally/Downloads/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
        "/Users/simhadrikondapally/Downloads/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"
    ]

    dataframes = []

    for path in csv_files:
        if not os.path.exists(path):
            st.error(f"File not found: {path}")
            continue
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            st.success(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")

    if not dataframes:
        st.error("No CSV files could be loaded.")
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = load_data()

if df.empty:
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# FIX CIC-IDS2017 COLUMN ISSUES
# --------------------------------------------------
df.columns = df.columns.str.strip()
st.info("Column names normalized")

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
st.subheader("🧹 Data Preprocessing")

df.drop_duplicates(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# --------------------------------------------------
# Detect Label Column Robustly
# --------------------------------------------------
label_col = None
for col in df.columns:
    if col.lower() == "label":
        label_col = col
        break

if label_col is None:
    st.error("Label column not found in dataset")
    st.stop()

# Binary encoding
df[label_col] = df[label_col].astype(str)
df[label_col] = df[label_col].apply(
    lambda x: 0 if "BENIGN" in x.upper() else 1
)

st.success("Data cleaned and labels encoded")

# --------------------------------------------------
# Feature & Target Split
# --------------------------------------------------
X = df.drop(label_col, axis=1)
y = df[label_col]

X = X.select_dtypes(include=[np.number])

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
st.subheader("🤖 Model Training")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
st.success("Random Forest model trained successfully")

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.subheader("📈 Model Evaluation")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy * 100:.2f}%")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# Live Prediction (From Real Data)
# --------------------------------------------------
st.subheader("🚦 Live Prediction (Real Traffic Sample)")

sample_input = X_test.iloc[:1]
prediction = model.predict(sample_input)[0]

if prediction == 0:
    st.success("🟢 NORMAL (BENIGN Traffic)")
else:
    st.error("🔴 MALICIOUS (ATTACK Detected)")

st.write("Sample Record Used:")
st.dataframe(sample_input)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "**AI-Based Network Intrusion Detection System**  |  "
    "CIC-IDS2017 Dataset  |  Random Forest  |  Streamlit"
)