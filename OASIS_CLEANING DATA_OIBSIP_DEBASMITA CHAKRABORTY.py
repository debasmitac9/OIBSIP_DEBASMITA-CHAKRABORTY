#!/usr/bin/env python
# coding: utf-8

# In[55]:


# importing necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('AB_NYC_2019.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler


# In[3]:


# to read .csv file of dataset

df=pd.read_csv("AB_NYC_2019.csv")
df.head(10)


# In[4]:


# Exploratory Data Analysis (EDA) to identify number of null

#Basic EDA
df.isnull().sum()


# In[5]:


# descriptive analysis to summarize data

df.describe()


# In[6]:


# Data Cleaning : by removing / dropping non-numeric columns

non_numeric_columns=df.select_dtypes(exclude=[np.number]).columns
df_numeric=df.drop(columns=non_numeric_columns,axis=1)


# In[9]:


# Distribution of different room types

sns.countplot(x='room_type',data=df)
plt.title("Room Types Distribution")
plt.show()


# In[12]:


# Distributiion of prices

plt.figure(figsize=(10,6))
sns.histplot(df['price'],bins=100,kde=True)
plt.title("Prices Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# In[32]:


print("Skewness: %f" % df['price'].skew())
print("Kurtosis: %f" % df['price'].kurt())

# as per skewness and kurtosis, we can say that the distribution is not normal or it doesn't provide equal distribution


# In[41]:


# Outliers detectios

plt.figure(figsize=(6,4))
sns.boxplot(data=df, y='price')
plt.title("Box Plot of Prices Distribution")
plt.ylabel("Price")
plt.show()


# In[45]:


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
outliers.head()


# In[46]:


#Correlation matrix
correlation_matrix=df_numeric.corr()
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


# # Number of Reviews with Review per Month shows positive correlation with the specific value of 0.55

# # Feature Engineering

# In[47]:


# desc analysis of summarised information of Price
df.price.describe()


# In[48]:


#Setting the min and max thresold for cleaning the data
minThresold,maxThresold=df.price.quantile([0.01,0.999])
minThresold,maxThresold


# In[49]:


#Datapoints where price is less than minThresold (1 percentile) and greater than the maxThresold (99 percentile)

df1=df[(df.price<minThresold)|(df.price>maxThresold)]
df1.shape


# In[50]:


df1.head(10)


# In[51]:


# Price distribution within min- and max- Thresold

df2=df[(df.price>minThresold)&(df.price<maxThresold)]
df2.shape #Datapoints where prices is in between the min and max thresold  (data points where price is lie between 1 percentile to 99 percentile)


# In[52]:


df2.head(10)


# In[53]:


# descriptive analysis to summarize data of Price

df2.price.describe()


# In[54]:


# Outliers removal

data = df[(df['price'] > lower_bound) & (df['price'] < upper_bound)]
data.describe()


# In[ ]:




