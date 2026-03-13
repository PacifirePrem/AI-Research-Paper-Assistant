# Credit Card Fraud Detection System

This project builds a machine learning pipeline to detect fraudulent credit card transactions using the Credit Card Fraud dataset.

## Features

- Data preprocessing and scaling
- Handling imbalanced data using SMOTE
- Random Forest fraud detection model
- Model evaluation using Precision, Recall, F1-score
- Interactive Streamlit dashboard for fraud prediction


## How to Run

Install dependencies

pip install -r requirements.txt

Train the model

python src/models/train_model.py

Run the dashboard

python -m streamlit run app/app.py

## Tech Stack

Python  
Scikit-learn  
Pandas  
Streamlit  
Imbalanced-learn  
Random Forest

## Example Output

The model detects fraudulent transactions and displays them through an interactive dashboard.

## Author

Prem Kumar


