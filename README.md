# Python-project-CREDIT-RISK-ANALYSIS.py

📌 Project Description – Credit Risk Analysis

This project focuses on analyzing credit risk using data-driven techniques to understand borrower behavior and identify customers who are more likely to default on loans. The analysis is performed on a structured dataset containing demographic details, financial information, loan characteristics, and repayment history.

The project begins with loading the dataset and performing an initial inspection to understand the overall structure, data types, distribution, and missing values. All missing values—especially in key fields like EducationLevel—are handled appropriately, and duplicate records are removed to ensure clean and reliable data.

Next, feature engineering techniques are applied to enhance the dataset. A new feature called Debt-to-Income Ratio is created to evaluate financial stability, and Credit History is categorized into Short, Medium, and Long periods. Categorical variables such as Gender, Employment Status, Property Area, and Marital Status are encoded using Label Encoding for compatibility with modeling techniques.

Extensive Exploratory Data Analysis (EDA) is conducted to understand the major factors influencing loan repayment. Visualizations include repayment distribution, age-wise default risk, impact of credit history, and correlations among numerical attributes. These insights reveal patterns that help identify financial and demographic indicators of risk.

For customer segmentation, K-Means Clustering is used after scaling numerical features with StandardScaler. The Elbow Method helps determine the optimal number of clusters, and customers are grouped into three Risk Categories: Low, Medium, and High. Each cluster represents a distinct group of borrowers with similar financial characteristics and default tendencies.

The project includes visualizations that show the distribution of customers across risk categories and how repayment behavior varies within each cluster. These insights enable financial institutions to design better loan approval strategies, reduce default rates, and improve portfolio quality.

This project provides a strong base for future enhancements, such as predictive modeling, risk scoring, or deploying machine learning models for real-time credit risk assessment.
