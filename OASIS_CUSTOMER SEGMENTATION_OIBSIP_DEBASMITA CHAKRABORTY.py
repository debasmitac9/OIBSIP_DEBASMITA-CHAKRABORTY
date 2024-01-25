#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Customer segmentation is a powerful marketing technique that involves dividing a customer base into distinct segments based on shared characteristics, behaviours, or demographics. The primary purpose of customer segmentation is to better understand and serve customers in a more personalized and targeted way. Marketing segmentation helps to understand customer needs better and reach the right customer with right messaging.
# Exploratory Data Analysis (EDA) is a necessary preliminary step before using a segmentation algorithm.

# # Data
# The data contains 2,205 observations and 39 columns. 

# # Data Preparation and Cleaning
# In this section, the following operations are performed:
# 
# 1. Reviewing data columns and comparing them to the dataset description
# 2. Looking for missing values
# 3. Checking column types
# 4. Assessing unique values

# In[1]:


#Importing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr


# In[2]:


#Reading the data
data = pd.read_csv('ifood_df.csv')

#Taking a look at the top 5 rows of the data
data.head()


# # Reviewing data columns and comparing them to the dataset description
# Retrieving the list of actual columns to compare with the column's description in the dictionary.png. The list of columns has been updated and the data description contains actual columns.

# In[3]:


data.columns


# # Looking for missing values
# Surprisingly, there is no missing values in the data and there are 2,205 observations in the data frame.

# In[4]:


data.isna().sum()


# # Checking column types
# All column types look good. There is no need to change any data types.

# In[5]:


data.info()


# # Assessing unique values
# Let's check the unique values in each column. If a column has the same values then we cannot use this column in our analysis and can remove it from the data frame.

# In[6]:


data.nunique()


# In[7]:


#Columns Z_CostContact and Z_Revenue have all the same values. These columns will not help us to understand our customers better. We can drop these columns from the data frame.


# In[8]:


data.drop(columns=['Z_CostContact','Z_Revenue'],inplace=True)


# # Data Exploration
# In this section:
# 
# Box plot for the total amount spent on all products (MntTotal)
# Outliers
# Box plot and histogram for income
# Histogram for age
# Correlation matrix
# Point-Biserial correlations for binary variables

# # Box plot for the total amount spent on all products (MntTotal)
# Our analysis will be focused on total amount spent on all products (MntTotal). Boxplot will help us to find outliers if any.

# In[9]:


plt.figure(figsize=(6, 4))  
sns.boxplot(data=data, y='MntTotal')
plt.title('Box Plot for MntTotal')
plt.ylabel('MntTotal')
plt.show()


# # Outliers
# The box plot spotted a few outliers in the MntTotal. Let's take a closer look at the outliers.

# In[10]:


Q1 = data['MntTotal'].quantile(0.25)
Q3 = data['MntTotal'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data['MntTotal'] < lower_bound) | (data['MntTotal'] > upper_bound)]
outliers.head()


# In[11]:


#Outliers removal
data = data[(data['MntTotal'] > lower_bound) & (data['MntTotal'] < upper_bound)]
data.describe()


# # Box plot and histogram for income

# In[12]:


plt.figure(figsize=(6, 4))  
sns.boxplot(data=data, y='Income', palette='viridis')
plt.title('Box Plot for Income')
plt.ylabel('Income')
plt.show()


# In[14]:


#histogram for income
plt.figure(figsize=(8, 6))  
sns.histplot(data=data, x='Income', bins=20, kde=True)
plt.title('Histogram for Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()


# # Income distribution is close to normal distribution with no outliers.

# In[15]:


#histogram for age
plt.figure(figsize=(8, 6))  
sns.histplot(data=data, x='Age', bins=20, kde=True)
plt.title('Histogram for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[16]:


print("Skewness: %f" % data['Age'].skew())
print("Kurtosis: %f" % data['Age'].kurt())


# # The age distribution looks approximately symmetrical and the left and right sides of distribution are roughly equal. Skewness of 0.09 (close to zero) supports the visual observation of the distribution. Kurtosis of -0.8 suggests that the distribution is close to normal with lighter tails and less peaked than a normal distribution.

# # Correlation matrix
# There are many columns in the data. The correlation matrix will be very crowded if we use all columns of the data frame. We will group the columns and explore correlation between columns in each group and the column 'MntTotal'. We will focus on the column 'MntTotal' to understand how we can segment the customers who buy the most in overall. We can run similar analysis for every type of product.

# In[17]:


cols_demographics = ['Income','Age']
cols_children = ['Kidhome', 'Teenhome']
cols_marital = ['marital_Divorced', 'marital_Married','marital_Single', 'marital_Together', 'marital_Widow']
cols_mnt = ['MntTotal', 'MntRegularProds','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
cols_communication = ['Complain', 'Response', 'Customer_Days']
cols_campaigns = ['AcceptedCmpOverall', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
cols_source_of_purchase = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
cols_education = ['education_2n Cycle', 'education_Basic', 'education_Graduation', 'education_Master', 'education_PhD']


# In[18]:


#correlation matrix
corr_matrix = data[['MntTotal']+cols_demographics+cols_children].corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# # Here, MntTotal has strong positive correlation with income and intermediate negative correlation with Kidhome. Income feature has nearly the same negative correlation with Kidhome and MntTotal.

# # Point-Biserial correlations for binary variables
# Pearson correlation measures the strength and direction of a linear relationship between two continuous variables. We used Pearson correlation for MntTotal, Age and Income. When we try to understand the relationship between a continuous variable MntTotal and binary variables like marital status then we should use Point-Biserial Correlation Point-Biserial Correlation is used to measure the strength and direction of the linear relationship between a binary variable and a continuous variable.

# In[19]:


for col in cols_marital:
    correlation, p_value = pointbiserialr(data[col], data['MntTotal'])
    print(f'{correlation:.4f}: Point-Biserial Correlation for {col} with p-value {p_value:.4f}')   


# # There is no strong Point-Biserial correlation between MntTotal and different marital statuses. Some feature engineering may be required during the modelling process.

# In[20]:


for col in cols_education:
    correlation, p_value = pointbiserialr(data[col], data['MntTotal'])
    print(f'{correlation:.4f}: Point-Biserial Correlation for {col} with p-value {p_value:.4f}')  


# # There is no strong Point-Biserial correlation between MntTotal and various education levels.

# # Feature Engineering
# In this section:
# 
# 1. New feature: Marital
# 2. New feature: In_relationship

# # New feature: Marital
# The data frame contains 5 columns to reflect marital status. We are going to create a new column 'marital' with values: Divorced, Married, Single, Together, Widow. This column will allow us to draw some additional plots.

# In[21]:


def get_marital_status(row):
    if row['marital_Divorced'] == 1:
        return 'Divorced'
    elif row['marital_Married'] == 1:
        return 'Married'
    elif row['marital_Single'] == 1:
        return 'Single'
    elif row['marital_Together'] == 1:
        return 'Together'
    elif row['marital_Widow'] == 1:
        return 'Widow'
    else:
        return 'Unknown'
data['Marital'] = data.apply(get_marital_status, axis=1)


# In[22]:


plt.figure(figsize=(8, 6))
sns.barplot(x='Marital', y='MntTotal', data=data, palette='viridis')
plt.title('MntTotal by marital status')
plt.xlabel('Marital status')
plt.ylabel('MntTotal')


# # New feature: In_relationship
# There are 3 features that reflect if a person is single (Single, Divorced, Widow) and 2 features if a person is in relationship (Together, Married). We will add an additional feature 'In_relationship'. This feature will equal 1 if a customer's marital status is 'Married' or 'Together' and 0 in all other cases.

# In[23]:


def get_relationship(row):
    if row['marital_Married'] ==1:
        return 1
    elif row['marital_Together'] == 1:
        return 1
    else:
        return 0
data['In_relationship'] = data.apply(get_relationship, axis=1)
data.head() 


# # K-Means Clustering
# K-means clustering is an unsupervised machine learning algorithm used to cluster data based on similarity. K-means clustering usually works well in practice and scales well to the large datasets.
# 
# In this section:
# 
# 1. Standardising data
# 2. Principal Component Analysis (PCA)
# 3. Elbow method
# 4. Silhouette score analysis

# In[24]:


from sklearn.cluster import KMeans


# # Standardising data
# K-means clustering algorithm is based on the calculation of distances between data points to form clusters. When features have different scales, features with larger scales can disproportionately influence the distance calculation. There are various ways to standardise features, we will use standard scaling .

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols_for_clustering = ['Income', 'MntTotal', 'In_relationship']
data_scaled = data.copy()
data_scaled[cols_for_clustering] = scaler.fit_transform(data[cols_for_clustering])
data_scaled[cols_for_clustering].describe()


# # The mean value for all colums is almost zero and the standard deviation is almost 1. All the data points were replaced by their z-scores.

# # Principal Component Analysis (PCA)
# PCA is a technique of dimensionality reduction. PCA takes the original features (dimensions) and create new features that capture the most variance of the data.

# In[26]:


from sklearn import decomposition
pca = decomposition.PCA(n_components = 2)
pca_res = pca.fit_transform(data_scaled[cols_for_clustering])
data_scaled['pc1'] = pca_res[:,0]
data_scaled['pc2'] = pca_res[:,1]


# # Elbow method
# The elbow method is a technique used to determine the optimal number of clusters (K) for K-means clustering algorithm.

# In[27]:


X = data_scaled[cols_for_clustering]
inertia_list = []
for K in range(2,10):
    inertia = KMeans(n_clusters=K, random_state=7).fit(X).inertia_
    inertia_list.append(inertia)


# In[28]:


plt.figure(figsize=[7,5])
plt.plot(range(2,10), inertia_list, color=(54 / 255, 113 / 255, 130 / 255))
plt.title("Inertia vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()


# # Elbow method suggests 4 or 5 clusters. Let's check silhouette score.

# # Silhouette score analysis
# Silhouette score is a metric that used to assess the quality of clustering. A higher silhouette score indicates that the clusters are well-separated, while a lower score suggests that the clusters may overlap or are poorly defined.

# In[29]:


from sklearn.metrics import silhouette_score
silhouette_list = []
for K in range(2,10):
    model = KMeans(n_clusters = K, random_state=7)
    clusters = model.fit_predict(X)
    s_avg = silhouette_score(X, clusters)
    silhouette_list.append(s_avg)

plt.figure(figsize=[7,5])
plt.plot(range(2,10), silhouette_list, color=(54 / 255, 113 / 255, 130 / 255))
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()


# # The highest silhouette score is for 4 clusters.

# In[30]:


model = KMeans(n_clusters=4, random_state = 7)
model.fit(data_scaled[cols_for_clustering])
data_scaled['Cluster'] = model.predict(data_scaled[cols_for_clustering])


# # Exploration of Clusters
# In this section:
# 
# 1. Visualisation of clusters
# 2. Mean consumption of different product types by cluster
# 3. Cluster sizes
# 4. Income by cluster
# 5. In_relationship feature by cluster

# # Visualisation of clusters

# In[31]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='pc1', y='pc2', data=data_scaled, hue='Cluster', palette='viridis')
plt.title('Clustered Data Visualization')
plt.xlabel('Principal Component 1 (pc1)')
plt.ylabel('Principal Component 2 (pc2)')
plt.legend(title='Clusters')


# In[32]:


data['Cluster'] = data_scaled.Cluster
data.groupby('Cluster')[cols_for_clustering].mean()


# # Mean consumption of different product types by cluster

# In[33]:


mnt_data = data.groupby('Cluster')[cols_mnt].mean().reset_index()
mnt_data.head()


# In[34]:


melted_data = pd.melt(mnt_data, id_vars="Cluster", var_name="Product", value_name="Consumption")
plt.figure(figsize=(12, 6))
sns.barplot(x="Cluster", y="Consumption", hue="Product", data=melted_data, ci=None, palette="viridis")
plt.title("Product Consumption by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Product Consumption")
plt.xticks(rotation=0)  
plt.legend(title="Product", loc="upper right")

plt.show()


# # Cluster sizes

# In[35]:


cluster_sizes = data.groupby('Cluster')[['MntTotal']].count().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(x='Cluster', y='MntTotal', data=cluster_sizes, palette = 'viridis')
plt.title('Cluster sizes')
plt.xlabel('Cluster')
plt.ylabel('MntTotal')


# In[36]:


total_rows = len(data)
cluster_sizes['Share%'] = round(cluster_sizes['MntTotal'] / total_rows*100,0)
cluster_sizes.head()


# # Income by cluster

# # Box plot

# In[37]:


#Data Visualization
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Income', data=data, palette='viridis')
plt.title('Income by cluster')
plt.xlabel('Cluster')
plt.ylabel('Income')
plt.legend(title='Clusters')


# # Scatter plot

# In[38]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='Income', y='MntTotal', data=data, hue = 'Cluster', palette='viridis')
plt.title('Income by cluster')
plt.xlabel('Income')
plt.ylabel('MntTotal')
plt.legend(title='Clusters')


# # In_relationship feature by cluster

# In[43]:


plt.figure(figsize=(6,6))
sns.barplot(x='Cluster', y='In_relationship', data=data, palette='viridis')
plt.title('In_relationship by cluster')
plt.xlabel('Cluster')
plt.ylabel('In_relationship')


# # Results
# This section contains the results of the K-means clustering analysis, which aimed to identify distinct customer segments based on the total amount of purchases they made (MntTotal). The analysis utilised 'Income' and 'In_relationship' features.
# 
# # Optimal number of clusters = 4
# The Elbow Method and Silhouette Analysis suggested 4 clusters (k=4). The elbow method highlighted the number of 4 or 5 clusters as a reasonable number of clusters. The silhouette score analysis revealed a peak silhouette score for k=4.

# # Cluster Characteristics
# #Cluster 0: High value customers in relationship (either married or together)
# 1. This cluster represents 26% of the customer base
# 2. These customers have high income and they are in a relationship
# #Cluster 1: Low value single customers
# 1. This cluster represents 21% of the customer base
# 2. These customers have low income and they are single
# #Cluster 2: High value single customers
# 1. This cluster represents 15% of the customer base
# 2. These customers have high income and they are single
# #Cluster 3: Low value customers in relationship
# 1. This cluster represents 39% of the customer base
# 2. These customers have low income and they are in a relationship

# # Recommendations
# Based on the clusters, tailored marketing strategies can be created. Customers from these segments will have different interests and product preferences.
# 
# #Marketing Strategies for Each Cluster
# Cluster 0: High value customers in relationship (either married or together)
# 1. Preliminary analysis showed that high income customers buy more wines and fruits.
# 2. A tailored campaign to promote high quality wines may bring good results.
# 3. This cluster contains customers in relationship, family-oriented promo-images should be quite effective for this audience.

# #Cluster 1: Low value single customers
# 1. Promos with discounts and coupons may bring good results for this targeted group.
# 2. Loyalty program may stimulate these customers to purchase more often.
# 
# #Cluster 2: High value single customers
# 1. Similar to the Cluster 0, these customers buy a lot of wines and fruits.
# 2. This cluster contains single customers. Promo images with friends, parties or single trips may be more efficient for single customers
# 
# #Cluster 3: Low value customers in relationship
# 1. This cluster has the highest percentage of our customers (39%).
# 2. Family offers and discounts may influence these customers to make more purchases

# # Opportunities for the further analysis
# 
# 1. Further exploration on how children influence on the consumed products
# 2. Further analysis on the influence of education analysis of frequent buyers
# 3. Analysis of sales channels, e.g. store, website, etc.
# 4. Analysis of the response to the marketing campaigns
# 5. It would be great to add gender data to the dataset
# 6. Test different clustering algorithms

# # This is the overal project of Customer Segmentation
