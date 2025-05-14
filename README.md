Customer Conversion Prediction in Digital Marketing

Overview
This project focuses on predicting customer conversion using machine learning models based on digital marketing campaign data. It helps marketers identify key factors that influence conversion rates and optimize campaign strategies for better ROI.

Objectives
Analyze and clean marketing campaign data

Explore features influencing customer conversion

Apply machine learning models to predict conversion

Compare model performance to choose the best one

Use insights to guide marketing decision-making

Dataset
The dataset includes customer details, marketing channel, campaign type, ad spend, engagement metrics, and whether the customer converted (binary outcome).

Techniques Used
Exploratory Data Analysis (EDA)

Feature Encoding and Scaling

SMOTE for handling class imbalance

Machine Learning Models:

Logistic Regression

Decision Tree

Random Forest

XGBoost (Tuned via GridSearchCV)

Model Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Results
Tuned XGBoost outperformed other models with:

Accuracy: 95.83%

Precision: 94.20%

Recall: 97.83%

F1-score: 95.98%

ROC-AUC: 99%

Conclusion
The project helps marketers understand what factors drive conversions and supports data-driven strategies to improve campaign effectiveness.

📊 Dataset Used:

The dataset used in this project contains information related to digital marketing campaigns and customer interactions. It includes the following types of features:

🔹 Features:

Age – Customer's age

Gender – Male or Female

Campaign Type – Type of marketing campaign

Marketing Channel – Channel used (e.g., email, social media, etc.)

Ad Spend – Amount spent on each campaign

Previous Purchases – Number of past purchases by the customer

Loyalty Points – Customer's loyalty score

Conversion – Target variable (1 = Converted, 0 = Not Converted)

🔹 Target Variable:
Conversion – A binary variable indicating whether the customer responded positively to the marketing campaign.

The dataset was cleaned, preprocessed, and used for both exploratory data analysis and model building.



