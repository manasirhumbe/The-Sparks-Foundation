#!/usr/bin/env python
# coding: utf-8

# # TASK 1 - Prediction using Supervised ML
# 
# # To Predict the percentage of marks of the students based on the number of study hours.

# In[1]:


## Importing important libraries---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Importing Dataset-
path =  "http://bit.ly/w-data"
Data = pd.read_csv(path)
print("Data is successfully imported")
Data


# In[3]:


## Now print the first 5 records...

Data.head()


# In[4]:


## Now print the last 5 records...
Data.tail()


# In[5]:


## Here we use describe() method so that we can able to see percentiles,mean,std,max,count of the given dataset.
Data.describe()


# In[6]:


# Let's print the full summary of the dataframe .
Data.info()


# # Visualizing Data.

# In[7]:


## Ploting Scatter plot----
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Scores',fontsize=15)
plt.title('Hours studied vs Score', fontsize=20)
plt.scatter(Data.Hours,Data.Scores,color='blue',marker='*')
plt.show()


# # This "SCATTER PLOT" indicates positive linear relationship as much as hours You study is a chance of high scoring.

# In[8]:


X = Data.iloc[:,:-1].values
Y = Data.iloc[:,1].values
X


# In[9]:


Y


# # Preparing Data and splitting into train and test sets.

# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size=0.2)


# In[11]:


## We have Splitted Our Data Using 80:20 RULe(PARETO)
print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape  =", X_test.shape)
print("Y test.shape  =", Y_test.shape)


# # Training the Model.

# In[12]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[13]:


## Fitting Training Data
linreg.fit(X_train,Y_train)
print("Training our algorithm is finished")


# In[14]:


print("B0 =",linreg.intercept_,"\nB1 =",linreg.coef_)## β0 is Intercept & Slope of the line is β1.,"


# In[15]:


## Plotting the REGRESSION LINE---
Y0 = linreg.intercept_ + linreg.coef_*X_train


# In[16]:


## Plotting on train data
plt.scatter(X_train,Y_train,color='green',marker='+')
plt.plot(X_train,Y0,color='orange')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Train set)",fontsize=10)
plt.show()


# # Test Data.

# In[17]:


Y_pred=linreg.predict(X_test)##predicting the Scores for test data
print(Y_pred)


# In[18]:


## Now print the Y_test.
Y_test


# In[19]:


## Plotting line on test data
plt.plot(X_test,Y_pred,color='red')
plt.scatter(X_test,Y_test,color='black',marker='+')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=20)
plt.show()


# # Comparing Actual vs Predicted Scores.¶

# In[20]:


Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare


# # ACCURACY OF THE MODEL.¶

# In[21]:


from sklearn import metrics
metrics.r2_score(Y_test,Y_pred)##Goodness of fit Test


# # Above 94% percentage indicates that above fitted Model is a GOOD MODEL.

# 
# 
# 
# # # Predicting the Error

# In[22]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[23]:


MSE = metrics.mean_squared_error(Y_test,Y_pred)
root_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Squared Error      = ",MSE)
print("Root Mean Squared Error = ",root_E)
print("Mean Absolute Error     = ",Abs_E)


# # Predicting the score¶

# In[24]:


Prediction_score = linreg.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :",Prediction_score)


# # CONCLUSION:

# # From the above result we can say that if a studied for 9.25 then student will secured 93.69 MARKS.

# # Completed task#1
