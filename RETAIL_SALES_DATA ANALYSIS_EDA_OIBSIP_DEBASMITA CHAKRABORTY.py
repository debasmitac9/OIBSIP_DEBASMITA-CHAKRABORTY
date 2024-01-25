#!/usr/bin/env python
# coding: utf-8

# In[41]:


#Importing Modules
import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns # data processing


# In[4]:


import os
for dirname, _, filenames in os.walk('retail_sales_dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


sales_df=pd.read_csv("retail_sales_dataset.csv")


# In[7]:


#Exploratory Data Analysis (EDA)
# Display the first few rows of the dataset
sales_df.head()


# In[8]:


# Display the first few rows of the dataset
sales_df.tail()


# In[9]:


sales_df.info()


# In[10]:


# Summary statistics of desc analysis
sales_df.describe()


# In[16]:


#Data Visualization
# Distribution of customer age
plt.figure(figsize=(8, 6))
plt.hist(sales_df['Age'], bins=20, color='pink', edgecolor='green')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age of the Customer')
plt.show()


# In[28]:


#Time Series Analysis
#Sales over time
plt.figure(figsize=(10, 5))
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
monthly_sales = sales_df.groupby(sales_df['Date'].dt.to_period('M'))['Total Amount'].sum()
monthly_sales.plot(kind='line', marker='*')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)
plt.show()


# In[31]:


# Product category distribution
plt.figure(figsize=(8, 6))
product_counts = sales_df['Product Category'].value_counts()
product_counts.plot(kind='bar', color='deeppink', edgecolor='green')
plt.xlabel('Product Category')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Product Categories')
plt.xticks(rotation=45)
plt.show()


# In[34]:


# Correlation Analysis using Scatterplot in between age and total spending
plt.figure(figsize=(8, 6))
plt.scatter(sales_df['Age'], sales_df['Total Amount'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Total Amount')
plt.title('Correlation between Age and Total Spending')
plt.show()


# In[42]:


# Data Cleaning
sales_df=sales_df.drop(columns=['Date','Customer ID','Gender','Product Category'])
print('Modified DataFrame:\n',sales_df)


# In[43]:


# Calculate correlation matrix
correlation_matrix = sales_df.corr()
sales_df.corr()


# In[44]:


# Data Visualization
# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()



# In[ ]:


#Process to read Heatmap
#Look at the color of each cell to see the strength and direction of the correlation.
#Darker colors indicate stronger correlations, while lighter colors indicate weaker correlations.
#Positive correlations (when one variable increases, the other variable tends to increase) are usually represented by warm colors, such as red or orange.
#Negative correlations (when one variable increases, the other variable tends to decrease) are usually represented by cool colors, such as blue or green.

#Here, highest correlation between 'Total Amount' and 'Price per Unit' is present with 0.85 value, by following the next correlation between 'Total Amount' and 'Quantity' with 0.37 value etc. So, it proves that 'Price per Unit' is increased with 'Total Amount', similarly, 'Quantity' is increased with 'Total Amount' but 'Age' is negatively correlated with 'Quantity', 'Transaction ID', 'Price per Unit', 'Total Amount'. 

