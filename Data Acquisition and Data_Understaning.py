#!/usr/bin/env python
# coding: utf-8

# <h1 id="data_acquisition">Data Acquisition</h1>
#     <li>data source: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data" target="_blank">https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data</a></li>
#     <li>data type: csv</li>
# 

# In[1]:


# import pandas library
import pandas as pd


# <h2>Read Data</h2>
# 

# In[2]:


# Import pandas library
import pandas as pd

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)


# In[3]:


# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
df.head()


# In[4]:



df.tail(10)


# <h3>Add Headers</h3>
# 

# In[5]:



headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)


# In[6]:


df.columns = headers
df.head(10)


# we can drop missing values along the column "price" as follows  

# In[7]:


df.dropna(subset=["price"], axis=0)


# In[8]:



print(df.columns)


# <h2>Save Dataset</h2>
# 
df.to_csv("automobile.csv", index=False)
# <h1 id="basic_insight">Basic Insight of Dataset</h1>
# 

# <h2>Data Types</h2>
# 

# In[9]:


df.dtypes


# In[10]:


# check the data type of data frame "df" by .dtypes
print(df.dtypes)


# <h2>Describe</h2>
# 

# In[11]:


df.describe()


# In[12]:


# describe all the columns in "df" 
df.describe(include = "all")


# In[14]:


df.info()


# In[ ]:




