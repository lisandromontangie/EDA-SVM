#!/usr/bin/env python
# coding: utf-8


#######################################################
# Exploratory Data Analysis and Prediction model      #
# for the Bike Sharing Data Set ([1] Fanaee-T, Hadi,  #
# and Gama, Joao, "Event labeling combining ensemble  #
# detectors and background knowledge", Progress in    #
# Artificial Intelligence (2013): pp. 1-15, Springer  #
# Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.   #
#######################################################

# In[2]:

# Import relevant libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm


# In[3]:


# Load data from file
data = pd.read_csv(path+'/hour.csv')


# In[4]:


# Obtain a general description of data
data.info()
data.isnull().any()
data.describe()
data.head()


# In[5]:


# Correct categorical variables interpreted as numerical
data['season'] = data.season.astype('category')
data['mnth'] = data.mnth.astype('category')
data['hr'] = data.hr.astype('category')
data['holiday'] = data.holiday.astype('category')
data['weekday'] = data.weekday.astype('category')
data['workingday'] = data.workingday.astype('category')
data['weathersit'] = data.weathersit.astype('category')

#check
data.dtypes


# In[6]:


# Set aside test data
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# In[7]:


# Check user count distribution
f, ax = plt.subplots(figsize=(7.5,5))
sns.distplot(train_set['cnt'])
ax.set(title='Distribution of user count', xlabel='User count')
plt.savefig("userdist.png")


# In[8]:


# Check behaviour of user count on weekends vs work days
fig, ax = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['hr', 'cnt', 'weekday']], x='hr', y='cnt', hue='weekday', ax=ax)
ax.set(title='Hourly usage per day of the week', xlabel='Hour of the day', ylabel='User count')
plt.savefig("workvsends.png")


# In[9]:


# Check if particular days are relevant
plt.subplots(figsize=(7.5,5))
sns.barplot(data=train_set[['weekday', 'cnt']], x='weekday', y='cnt')
plt.xlabel('Weekday')
plt.ylabel('Daily user count')
plt.title('Usage during the week')
plt.savefig("weekday.png")


# In[10]:


# Comparison to 'workingday' feature 
fig, ax = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['hr', 'cnt', 'workingday']], x='hr', y='cnt', hue='workingday', ax=ax)
ax.set(title='Hourly usage (working vs non-working day)', xlabel='Hour of the day', ylabel='User count')
plt.savefig("workingday.png")


# In[11]:


# Check hourly behaviour on the month of the year
fig, ax = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['hr', 'cnt', 'mnth']], x='hr', y='cnt', hue='mnth', ax=ax)
ax.set(title="Use of the system depending on months")
plt.savefig("monthlybehaviour.png")


# In[12]:


# Check total usage dependency on month of the year
fig = plt.subplots(figsize=(7.5,5))
sns.barplot(data=train_set[['mnth', 'cnt']], x='mnth', y='cnt')
plt.xlabel('Month')
plt.ylabel('Monthly user count')
plt.title('Usage during the year')
plt.savefig("monthlytotal.png")


# In[13]:


# Check total usage vs seasons
fig = plt.subplots(figsize=(7.5,5))
sns.barplot(data=train_set[['season', 'cnt']], x='season', y='cnt')
plt.xlabel('Season')
plt.ylabel('User count per season')
plt.title('Usage during the season')
plt.savefig("seasontotal.png")


# In[14]:


# Check dependency of usage on temperature
fig = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['temp', 'cnt']], x='temp', y='cnt')
plt.xlabel('Normalized temperature')
plt.ylabel('User count')
plt.title('Usage vs temperature')
plt.savefig("temp.png")


# In[15]:


# Check dependency of usage on feeling temperature
fig = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['atemp', 'cnt']], x='atemp', y='cnt')
plt.xlabel('Normalized feeling temperature')
plt.ylabel('User count')
plt.title('Usage vs feeling temperature')
plt.savefig("atemp.png")


# In[16]:


# Check dependency on weather situation
fig = plt.subplots(figsize=(7.5,5))
sns.barplot(data=train_set[['weathersit', 'cnt']], x='weathersit', y='cnt')
plt.xlabel('Weather situation')
plt.ylabel('User count')
plt.title('Usage per weather situation')
plt.savefig("weathersit.png")


# In[17]:


# Check dependency on wind speed
fig = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['windspeed', 'cnt']], x='windspeed', y='cnt')
plt.xlabel('Windspeed')
plt.ylabel('User count')
plt.title('Usage vs windspeed')
plt.savefig("windspeed.png")


# In[18]:


# Check dependency on humidity
fig = plt.subplots(figsize=(7.5,5))
sns.pointplot(data=train_set[['hum', 'cnt']], x='hum', y='cnt')
plt.xlabel('Humidity')
plt.ylabel('User count')
plt.title('Usage vs humidity')
plt.savefig("humidity.png")


# In[19]:


# Check correlations 
train_set_corr = train_set.corr()
mask = np.array(train_set_corr)
mask[np.tril_indices_from(mask)] = False
fig = plt.subplots(figsize=(15,10))
sns.heatmap(train_set_corr, mask=mask, vmax=1, square=True, annot=True)
plt.savefig("correlations.png")


# In[20]:


# Drop non-relevant entries

train_set = train_set.drop(['instant','dteday','yr','casual','registered','weekday','holiday','atemp','windspeed'], axis=1)
test_set = test_set.drop(['instant','dteday','yr','casual','registered','weekday','holiday','atemp','windspeed'], axis=1)


# In[21]:


# Transform "cnt" distribution logarithmicaly

train_set['cnt'] = train_set['cnt'].transform(lambda x: math.log(x))
test_set['cnt'] = test_set['cnt'].transform(lambda x: math.log(x))

f, ax = plt.subplots(figsize=(7.5,5))
sns.distplot(train_set['cnt'] , color="red")
ax.set(title='Distribution of transformed user count', xlabel='User count')
plt.savefig("newuserdist.png")


# In[22]:


# Perform "one-hot encoding" for categorical variables
dummy_train= train_set
dummy_test= test_set

# Define function for posterior use in Pipeline if needed
def one_hot_encoding(dataframe, categorical_feature):       
    dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[categorical_feature], prefix=categorical_feature, drop_first=True)],axis=1)
    dataframe = dataframe.drop([categorical_feature], axis=1)
    return dataframe
 
categorical_features=['season','mnth','weathersit']
for feature in categorical_features:
    dummy_train = one_hot_encoding(dummy_train, feature)
    dummy_test = one_hot_encoding(dummy_test, feature)


# In[23]:


# Feature scaling (Standardization) for continous variables
    
def standarization(dataframe, continous_feature):       
    dataframe[continous_feature] = (dataframe[continous_feature]-dataframe[continous_feature].mean())/dataframe[continous_feature].std()
    return dataframe
 
numerical_features=['temp','hum']
for feature in numerical_features:
    dummy_train = standarization(dummy_train, feature)
    dummy_test = standarization(dummy_test, feature)


# In[24]:


# Train Support Vector Machine model (with a Radial Basis Function kernel) for regression 

y_train = dummy_train['cnt']
X_train = dummy_train.drop(['cnt'], axis=1)

X_test = dummy_test.drop(['cnt'], axis=1)
y_test = dummy_test['cnt']

model = svm.SVR(kernel='rbf')

model.fit(X_train, y_train.values.flatten())
y_pred = model.predict(X_test)

print("The test score is ", model.score(X_test,y_test.values.flatten()))
print("The Mean Absolute is ", metrics.mean_absolute_error(y_test.values.flatten(), y_pred))

# Plot the residuals
res = y_test-y_pred
fig, ax = plt.subplots(figsize=(7.5,5))
sns.scatterplot(y_test, res)
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text('Plot of Residuals')
plt.savefig("res.png")
plt.show()

