#!/usr/bin/env python
# coding: utf-8

# # Task-2: Prediction using Unsupervised Machine Learning
# 
# 
# 
# 
# From the given "Iris" dataset, predict the optimum number of clusters & represent it visually.

# In[1]:


##Importing libraries:
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the iris dataset

# In[2]:


df=pd.read_csv(r"C:\Users\hp\Desktop\Iris.csv")
df.drop(['Id'],axis=1,inplace=True)


# # Reading the dataset

# In[3]:


df.head() #diplay the fisrt 5 dataset.


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop_duplicates(inplace=True)


# # Label Encoding

# In[9]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
df['Species'].value_counts()


# # PetalLengthCm vs PetalWidthCm
# 
# • We'll compare our final plot with this graph to check how accurate our model is

# In[10]:


plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'],c=df.Species.values)


# In[11]:


df.corr()


# # Data Visualization

# In[12]:


fig=plt.figure(figsize=(15,12))
sns.heatmap(df.corr(),linewidths=1,annot=True)


# In[13]:


sns.pairplot(df)


# # 
# 
# • We can see that Species is mainly depend on Petal Length and Petal Width.
# 
# • Using petal_length and petal_width

# In[15]:


df=df.iloc[:,[0,1,2,3]].values


# # Elbow Method using within-cluster-sum-of-squares(wcss)

# In[16]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
    
wcss


# # Using Elbow graph to find optimum no. of Clusters

# In[17]:


plt.figure(figsize=(10,5))
sns.set(style='whitegrid')
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


The optimum value for K would be 3. As we can see that with an increase in the number of clusters the WCSS value decreases.
We select the value for K on the basis of the rate of decrease in WCSS and we can see that after 3 the drop in wcss is minimal.


# # Initialization using K-means++

# In[18]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 5)
y_kmeans = kmeans.fit_predict(df)
y_kmeans


# # Visualizing the Clusters

# In[19]:


fig = plt.figure(figsize=(10, 7))
plt.title('Clusters with Centroids',fontweight ='bold', fontsize=20)
plt.scatter(df[y_kmeans == 0, 2], df[y_kmeans == 0, 3], s = 100, c = 'seagreen', label = 'Iris-versicolour')
plt.scatter(df[y_kmeans == 1, 2], df[y_kmeans == 1, 3], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(df[y_kmeans == 2, 2], df[y_kmeans == 2, 3],s = 100, c = 'yellow', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 300, c = 'red',marker='*', 
            label = 'Centroids')
plt.title('Iris Flower Clusters')
plt.ylabel('Petal Width in cm')
plt.xlabel('Petal Length in cm')
plt.legend()


# # We can see that our predicted graph is quite similar to the actual one.

# In[ ]:




