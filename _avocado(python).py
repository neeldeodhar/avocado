#!/usr/bin/env python
# coding: utf-8

# In[10]:


#downloading dataset, importing libraries
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
#from sklearn import svm
from sklearn.svm import SVR


# In[11]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('avocado.csv')



df.head()

df.dropna()


# In[12]:


#printing columns
df.columns


# In[13]:


# encoding "type" and selecting features
enc = OrdinalEncoder()


df[['type']] = enc.fit_transform(df[['type']])


selected_features = df[['Total Volume','4046', '4225', 'Total Bags', 'Small Bags','Large Bags', 'XLarge Bags','type','year']]
display(selected_features)


# In[14]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X = scaler.fit_transform(selected_features)


# In[15]:


# defining y variable

y = df['AveragePrice'].values


# In[16]:


#creating a split (train/test)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=45)


#creating a split (train/val)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=45)

# now the train/validate/test split will be 80%/10%/10%


# In[17]:


#training KN regressor for testing
score = []
for i in range(1,100):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(X_train, y_train)
    outputs = knn.predict(X_test)
    
    score.append (mean_squared_error(y_test, outputs))


# In[18]:


#training KN regressor for validation
score_val = []
for i in range(1,100):
    knn_val = KNeighborsRegressor(n_neighbors = i)
    knn_val.fit(X_train, y_train)
    outputs_val = knn_val.predict(X_val)
    
    score_val.append (mean_squared_error(y_val, outputs_val))


# In[19]:


#plotting accuracy percentage vs n_neighbors


plt.plot(range(1,100), score)
plt.title("Accuracy percent vs: n_neighbors ")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy Rate")

plt.show()


# In[20]:


#printing R squared score (test)
knn.score(X_test, y_test)


# In[21]:


#printing R squared score (test)
knn.score(X_val, y_val)


# In[22]:


#visualization to determine average price per region
plt.figure(figsize =(10,11))
plt.title("Avg. Price of avocado by region")
Av = sns.barplot(x = "AveragePrice", y = "region", data = df)



# In[23]:


#average price of avocados by type
plt.figure(figsize =(5,7))
plt.title("Avg.Price of Avocados by type")
Av = sns.barplot(x="type", y= "AveragePrice", data = df)


# In[24]:


#training SVR Regressor model withmax iterations of 10,000

for i in ['linear', 'poly', 'rbf', 'sigmoid']:
    dat = SVR(kernel = i, max_iter = 10000)
    
    dat.fit(X_train, y_train)
    scores = dat.score(X_test, y_test)
    print (i, scores)


# In[26]:


#conclusion
print ("KNN regressor offer's higher accuracy as compared to the SVR Regressor. However, the 'rbf' kernel's accuracy is close enough")


# In[ ]:




