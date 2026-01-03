🛡️ AI-Based Network Intrusion Detection System

A machine learning–based Network Intrusion Detection System (NIDS) developed using the CIC-IDS-2017 benchmark dataset.
The system classifies network traffic as Normal or Intrusion using a Random Forest classifier and presents results through an interactive Streamlit web dashboard.

📌 Project Overview

Network Intrusion Detection Systems play a critical role in identifying malicious activities within a network. Traditional rule-based systems often fail to detect new or evolving attack patterns.
This project demonstrates how supervised machine learning can be applied to network flow data to effectively distinguish between benign and malicious traffic.

The application trains a classification model on labeled network traffic data and provides visual performance analysis along with a simulated real-time detection interface.

🎯 Objectives

To build an ML-based intrusion detection system using benchmark network traffic data

To classify traffic into Normal and Intrusion categories

To visualize model performance using standard evaluation metrics

To simulate real-time traffic classification in a controlled environment

To deploy the system as an interactive web application

🧠 Methodology

Dataset

CIC-IDS-2017 (Friday Afternoon DDoS traffic)

Network flow–based features with labeled attack and benign samples

Preprocessing

Removal of non-numeric features

Handling of missing and infinite values

Binary label encoding (BENIGN → Normal, Attack → Intrusion)

Model

Random Forest Classifier

Stratified train–test split (75% / 25%)

Evaluation

Accuracy and error rate

Confusion matrix

Classification report

ROC curve

Precision–Recall curve

Feature importance analysis

Deployment

Interactive dashboard using Streamlit

Dataset uploaded dynamically via browser for cloud compatibility

📊 Key Features

Machine learning–based intrusion detection

Interactive Streamlit dashboard

Confusion matrix and performance metrics

ROC and Precision–Recall curves

Feature importance visualization

Dataset-based live traffic simulation

Cloud-deployable architecture

🔴 Live Traffic Simulation

The system simulates real-time intrusion detection by randomly sampling unseen network flow records from the dataset and classifying them on the fly.
This approach emulates how a trained NIDS would analyze incoming traffic without performing direct packet capture.
🧪 Results Summary

Achieved very high detection accuracy on the selected dataset

Minimal false positives and false negatives

Strong separation between benign and attack traffic

Feature importance analysis highlights traffic volume and packet-level characteristics as key indicators

Note: High accuracy is expected due to the well-defined attack patterns present in the selected benchmark dataset.

🛠️ Tech Stack

Programming Language: Python

Machine Learning: Scikit-learn (Random Forest)

Web Framework: Streamlit

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

📂 Project Structure
AI-Based-Network-Intrusion-Detection-System/
│── nids_main.py
│── README.md
│── requirements.txt
│── .gitignore

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run nids_main.py

🚀 Future Scope

Extension to multi-class intrusion detection

Training on additional CIC-IDS-2017 attack datasets

Integration with real-time network traffic sources

Performance optimization for large-scale deployments

👤 Author

Sukrit Jha
Electrical Engineering, NSUT

📄 Disclaimer

This project is developed for academic and research purposes.
It does not perform real-time packet sniffing or active network monitoring.
