#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Analysis

# In[ ]:





# In[ ]:





# # Step 1: Import Necessary Libraries

# In[1]:


# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[3]:


# For visualizations
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Set the style for plots
sns.set(style="whitegrid")


# # Step 2: Load Dataset and Initial Inspection

# In[6]:


# Load the dataset
df = pd.read_csv("C:/Users/HP/Downloads/credit_risk_dataset.csv")


# In[7]:


# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())


# In[8]:


# Check for basic information and missing values
print("\nDataset Information:")
print(df.info())


# In[9]:


# Summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())


# # Step 3: Data Cleaning

# In[10]:


# Handling missing or inconsistent values
# Checking for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())


# In[11]:


# Since we have missing values in the 'EducationLevel' column, we need to handle them:
# Option 1: Fill missing values with the most common (mode) education level
df['EducationLevel'].fillna(df['EducationLevel'].mode()[0], inplace=True)


# In[12]:


# Checking again for missing values to confirm that they are handled
print("\nMissing Values after handling:")
print(df.isnull().sum())


# In[13]:


# Dropping any potential duplicate rows in the dataset
df.drop_duplicates(inplace=True)


# In[14]:


# Display the first few rows to verify the changes
print("\nFirst 5 rows after Data Cleaning:")
print(df.head())


# # Step 4: Feature Engineering

# In[15]:


# Feature Engineering
# Create a new feature: Debt-to-Income Ratio
df['DebtToIncomeRatio'] = (df['LoanAmount'] / df['Income']).round(2)


# In[16]:


# Binning 'CreditHistory' into categories: 'Short', 'Medium', 'Long'
bins = [0, 5, 15, 30]
labels = ['Short', 'Medium', 'Long']
df['CreditHistoryCategory'] = pd.cut(df['CreditHistory'], bins=bins, labels=labels)


# In[17]:


# Encode categorical variables for clustering
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['EducationLevel'] = label_encoder.fit_transform(df['EducationLevel'])
df['PropertyArea'] = label_encoder.fit_transform(df['PropertyArea'])
df['EmploymentStatus'] = label_encoder.fit_transform(df['EmploymentStatus'])
df['RepaymentStatus'] = label_encoder.fit_transform(df['RepaymentStatus'])


# In[18]:


# Display first few rows after feature engineering
print("\nData after Feature Engineering:")
print(df.head())


# # Step 5: Exploratory Data Analysis (EDA)

# In[19]:


# EDA - Loan Default Analysis
# Plot the distribution of 'RepaymentStatus'
plt.figure(figsize=(10, 5))
sns.countplot(x='RepaymentStatus', data=df)
plt.title('Distribution of Loan Repayment Status')
plt.xlabel('Repayment Status (0 = Paid, 1 = Default)')
plt.ylabel('Count')
plt.show()


# In[20]:


# EDA - Default Probability by Demographics (e.g., Age)
plt.figure(figsize=(12, 6))
sns.boxplot(x='RepaymentStatus', y='Age', data=df)
plt.title('Loan Default Probability by Age')
plt.xlabel('Repayment Status (0 = Paid, 1 = Default)')
plt.ylabel('Age')
plt.show()


# In[21]:


# EDA - Default Probability by Credit History Category
plt.figure(figsize=(10, 5))
sns.countplot(x='CreditHistoryCategory', hue='RepaymentStatus', data=df)
plt.title('Loan Default Probability by Credit History Category')
plt.xlabel('Credit History Category')
plt.ylabel('Count')
plt.show()


# In[23]:


# To calculate the correlation matrix, we need only numerical columns
# We'll exclude any non-numeric columns first

# Select only numerical columns for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])


# In[24]:


# Now, calculate the correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # Step 6: Customer Segmentation Using Clustering

# In[25]:


# Scaling numerical features for clustering
scaler = StandardScaler()
numerical_features = ['Age', 'Income', 'LoanAmount', 'LoanTerm', 'InterestRate', 'CreditScore', 'DebtToIncomeRatio']
scaled_features = scaler.fit_transform(df[numerical_features])


# In[26]:


# K-Means Clustering to segment customers into risk categories
# Determine the optimal number of clusters using the Elbow method
inertia = []
range_clusters = range(1, 11)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)


# In[27]:


# Plotting the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# In[28]:


# Choose an optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['RiskCategory'] = kmeans.fit_predict(scaled_features)


# In[29]:


# Map Risk Categories: 0 -> Low, 1 -> Medium, 2 -> High
risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
df['RiskCategory'] = df['RiskCategory'].map(risk_map)


# In[30]:


# Display the data with risk categories
print("\nData with Risk Categories:")
print(df[['CustomerID', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'RiskCategory']].head())


# # Step 7: Visualize Segmentation Results

# In[31]:


# Visualizing Risk Categories
plt.figure(figsize=(10, 6))
sns.countplot(x='RiskCategory', data=df, order=['Low', 'Medium', 'High'])
plt.title('Customer Segmentation by Risk Category')
plt.xlabel('Risk Category')
plt.ylabel('Count')
plt.show()


# In[32]:


# Default Probability by Risk Category
plt.figure(figsize=(12, 6))
sns.countplot(x='RiskCategory', hue='RepaymentStatus', data=df, order=['Low', 'Medium', 'High'])
plt.title('Default Probability by Risk Category')
plt.xlabel('Risk Category')
plt.ylabel('Count')
plt.show()


# In[ ]:




