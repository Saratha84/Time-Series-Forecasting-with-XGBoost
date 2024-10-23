#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install xgboost')
import xgboost as xgb


# In[2]:


df=pd.read_csv(r"H:\Python\time series\AEP_hourly.csv")
df=df.set_index('Datetime')
df.index=pd.to_datetime(df.index)


# In[3]:


print(df)


# In[4]:


color_pal = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
df.plot(style=".",figsize=(15,5),color=color_pal[0],title="Annual Energy Production in MW")


# In[5]:


train=df.loc[df.index<"01-01-2015"]
test=df.loc[df.index>="01-01-2015"]

fig,ax=plt.subplots(figsize=(15,5))
train.plot(ax=ax,label="Training")
test.plot(ax=ax,label="Testing")
ax.axvline("01-01-2015",color='black',ls="--")
ax.legend(["Training Set","Testing Set"])
plt.show()


# In[6]:


df.loc[(df.index>"01-01-2015")&(df.index<"01-08-2015")].plot()


# In[20]:


def create_features(df):
    df['hour']=df.index.hour
    df['dayofweek']=df.index.dayofweek
    df['quarter']=df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year
    df['dayofyear']=df.index.dayofyear
    return df


# In[21]:


df=create_features(df)


# In[22]:


fig,ax=plt.subplots(figsize=(10,8))
sns.boxplot(data=df,x='hour',y='AEP_MW')
ax.set_title("AEP by Hour")


# In[23]:


fig,ax=plt.subplots(figsize=(10,8))
sns.boxplot(data=df,x='month',y='AEP_MW')
ax.set_title("AEP by month")


# In[24]:


from sklearn.metrics import mean_squared_error


# In[17]:


train=create_features(train)
test=create_features(test)


# In[19]:


print(df.columns)


# In[25]:


FEATURES=['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
TARGET='AEP_MW'


# In[29]:


X_train=train[FEATURES]
y_train=train[TARGET]

X_test=test[FEATURES]
y_test=test[TARGET]


# In[32]:


reg=xgb.XGBRegressor(n_estimators=1000,early_stopping_rounds=50,learning_rate=0.01)
reg.fit(X_train,y_train,
        eval_set=[(X_train,y_train),(X_test,y_test)],
        verbose=100)


# In[33]:


reg.feature_importances_


# In[36]:


fi=pd.DataFrame(data=reg.feature_importances_,
               index=reg.feature_names_in_,
               columns=['importance'])


# In[37]:


print(fi)


# In[38]:


fi.sort_values('importance').plot(kind='barh',title='Feature Importance')
plt.show()


# In[46]:


test['prediction']=reg.predict(X_test)
df=df.merge(test[['prediction']],how='left',left_index=True,right_index=True)
ax=df[["AEP_MW"]].plot(figsize=(15,5))
df['prediction'].plot(ax=ax, style=".")
plt.legend(['Truth data','Predictions'])
ax.set_title("Raw Data and Predictions")
plt.show()


# In[47]:


np.sqrt(mean_squared_error(test['AEP_MW'],test['prediction']))


# In[ ]:




