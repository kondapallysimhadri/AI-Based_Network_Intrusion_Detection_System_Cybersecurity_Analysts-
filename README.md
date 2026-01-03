# Project Title
AI-Based Network Intrusion Detection System

# Student Details
Name: Simhadri Kondapally
Role: Data Science Student
Project Type: Machine Learning & Cybersecurity Project

# Project Description

This project implements an AI-Based Network Intrusion Detection System (NIDS) using Machine Learning techniques. The system analyzes real-world network traffic data from the CIC-IDS2017 dataset to classify traffic as either Benign or Malicious.

Multiple large-scale network traffic CSV files are loaded, preprocessed, and combined. A Random Forest classifier is trained to learn patterns of malicious behavior. The project is deployed using Streamlit, providing an interactive web interface for model training, evaluation, and real-time prediction.

# End Users

Network Administrators

Cybersecurity Analysts

Security Operations Center (SOC) Teams

Organizations managing enterprise networks

Students and researchers in cybersecurity

# Technologies Used

Programming Language: Python

Machine Learning Algorithm: Random Forest

Libraries: Pandas, NumPy, Scikit-learn

Dataset: CIC-IDS2017 (real-world network traffic data)

Web Framework: Streamlit

Tools: VS Code, PyEnv, GitHub

# Dataset Information

The project uses multiple CSV files from the CIC-IDS2017 dataset, including:

Monday-WorkingHours.pcap_ISCX.csv

Tuesday-WorkingHours.pcap_ISCX.csv

Wednesday-workingHours.pcap_ISCX.csv

Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv

Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

The system automatically merges all datasets into ONE dataframe.

# Project Workflow

Load multiple CIC-IDS2017 CSV files

Normalize column names and labels

Handle missing and infinite values

Perform feature and target split

Train a Random Forest classifier

Evaluate model performance

Perform live traffic prediction

Display results using Streamlit

# Results

Achieved ~99.96% accuracy on real network traffic data

High precision, recall, and F1-score

Successfully detected malicious traffic in real time

Efficient handling of large and imbalanced datasets

## How to Run
pip install -r requirements.txt
streamlit run nids_main.py
