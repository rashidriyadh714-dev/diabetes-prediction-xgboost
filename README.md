***Diabetes Risk Predictor***
**Executive Summary**

This project develops a machine learning-based web application that predicts an individual's risk of developing diabetes based on key health parameters. The application utilizes the XGBoost algorithm trained on medical data to provide risk assessments with 85% accuracy. The solution features a bilingual interface (English/Malay), interactive visualizations, and personalized prevention recommendations, making it accessible to diverse users in Malaysia's healthcare context.

**Problem Statement**

Diabetes is a growing health concern in Malaysia, with an estimated 3.9 million adults living with the condition. Early detection and prevention are crucial for managing diabetes, but many individuals lack access to convenient risk assessment tools. This project addresses this gap by providing an accessible, AI-powered tool that helps users understand their diabetes risk based on measurable health indicators, encouraging proactive health management.

**Dataset Source**

The model was trained on the Pima Indians Diabetes Database ( https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database )from the National Institute of Diabetes and Digestive and Kidney Diseases, supplemented with Malaysia-specific health data from the World Bank Open Data repository.

**Methodology**

Data Pipeline

**Data Collection**: Combined international diabetes data with Malaysia-specific health indicators
**Preprocessing**: Handled missing values, normalized features, and addressed class imbalance
**Feature Engineering**: Created clinically relevant features like BMI categories and risk scores
**Model Training**: Implemented and compared multiple algorithms (XGBoost, Random Forest, SVM, Logistic Regression)
**Hyperparameter Tuning**: Optimized using Bayesian optimization techniques
**Evaluation**: Assessed performance using AUC-ROC, precision, recall, and F1-score

Model Architecture

**Primary Algorithm**: XGBoost (Extreme Gradient Boosting)

**Baseline Model**: Logistic Regression for comparison

**Evaluation Metrics**: AUC-ROC (0.85), Accuracy (81.0%), F1-Score (0.730)

**Interpretability**: SHAP values for feature importance explanation

**Results**

Performance Metrics

Model   	          Accuracy	ROC AUC 	F1 Score	Precision	Recall

XGBoost	            85%	      0.85	    0.75	     0.72	    0.78

Random Forest	      82%	      0.82	    0.72	     0.70	    0.75

Logistic Regression	83%	      0.80	    0.70	     0.68     0.73

SVM                 70%       0.84      0.66       0.64     0.70


Key Findings

Glucose levels and BMI were the most significant predictors of diabetes risk.

The model achieved 85% AUC-ROC, indicating strong discriminatory power.

Feature importance analysis aligned with clinical understanding of diabetes risk factors.

**Demonstration**

Running the Streamlit App Locally

**Clone the repository:**
bash

git clone https://github.com/rashidriyadh714-dev/diabetes-prediction-xgboost

cd diabetes-risk-predictor

**Install dependencies:**

bash

pip install -r requirements.txt

**Run the application:**
bash

streamlit run app.py

Access the app at http://localhost:8501

Online Deployment

The application is deployed on Streamlit Community Cloud and accessible at:

Diabetes Risk Predictor App ( https://rashid-diabetes-prediction-xgboost.streamlit.app )

Application Features

Bilingual Interface: Toggle between English and Malay

Interactive Inputs: Sliders for health parameters with clinical ranges

Visual Risk Assessment: Color-coded risk visualization (green to red gradient)

Personalized Recommendations: Prevention tips based on risk level

Model Transparency: Performance metrics and feature importance display

Screenshots

<img width="1470" height="832" alt="Screenshot 2025-08-24 at 2 54 46 PM" src="https://github.com/user-attachments/assets/babd3f23-489f-47fe-bf21-56fc68caf35c" />


**Acknowledgement**

This project was developed as part of the BIT4333 Introduction to Machine Learning course at City University Malaysia. Special thanks to:

Sir Nazmirul Izzad Bin Nassir for project guidance and supervision

Kaggle and the National Institute of Diabetes and Digestive and Kidney Diseases for providing the dataset

Streamlit team for the excellent deployment platform

XGBoost developers for the powerful machine learning library

The open-source community for various Python libraries that made this project possible






