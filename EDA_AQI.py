#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statistics as s
import sklearn


# In[2]:


df=pd.read_csv('E:/Downloads/data-gov.csv')
df.head(40)


# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


a=df.iloc[0:2079,3:11]


# In[6]:


a


# In[7]:


b=a.iloc[0:2079,2:11]


# In[8]:


b


# In[9]:


b.loc[50]


# In[10]:


b.head(50)


# In[11]:


print(b.isnull())


# In[12]:


b.isnull().sum()


# In[13]:


b.dropna(subset=['pollutant_min','pollutant_max','pollutant_avg'], inplace=True)


# In[14]:


b


# In[15]:


b.isnull().sum()


# In[16]:


b.head(50)


# In[17]:


b.duplicated()


# In[18]:


b.duplicated().value_counts()


# In[19]:


fig, ax = plt.subplots(figsize=(10,10))
ax.scatter('pollutant_min', 'pollutant_max', c='pollutant_max', s=50, data=b)
ax.set_xlabel('Minimum Pollution')
ax.set_ylabel('Date');


# In[20]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=b["pollutant_min"])


# In[21]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=b["pollutant_max"])


# In[22]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=b["pollutant_avg"])


# In[23]:


d=b.iloc[:,1:5]


# In[24]:


d


# In[25]:


ax1 = sns.boxplot(x="pollutant_min", y="pollutant_max", data=b)


# In[26]:


var1 = s.variance(d["pollutant_avg"])
var2 = s.variance(d["pollutant_min"])
var3 = s.variance(d["pollutant_max"])


# In[27]:


sns.pairplot(d, height=4)


# In[28]:


e=d.iloc[:,1:4]
e


# In[29]:


g = sns.clustermap(e)


# In[30]:


fig, ax1 = plt.subplots(figsize=(10,10))
sns.violinplot('pollutant_min','pollutant_avg', data=e, orient='v')


# In[31]:


fig, ax1 = plt.subplots(figsize=(10,10))
sns.violinplot('pollutant_max','pollutant_avg', data=e, orient='v')
sns.violinplot('pollutant_min','pollutant_avg', data=e, orient='v')


# In[32]:


gd=e.groupby('pollutant_avg').agg(['mean','median'])
gd


# In[33]:


plt.plot(gd)


# In[34]:


sns.FacetGrid(d, hue="pollutant_id", height=5) .map(sns.distplot, "pollutant_avg") .add_legend()


# In[35]:


sns.FacetGrid(d, hue="pollutant_id", height=5) .map(sns.distplot, "pollutant_min") .add_legend()


# In[36]:


sns.FacetGrid(d, hue="pollutant_id", height=5) .map(sns.distplot, "pollutant_max") .add_legend()


# In[37]:


cr = df.iloc[:,:10]


# In[38]:


cr.corr()


# In[39]:


plt.figure(figsize=(10,9))
sns.heatmap(cr.corr(),cmap="vlag_r",annot=True,fmt='.2f',linewidths=2, cbar=False)
plt.title("Correlation Matrix")
plt.show()


# In[40]:


b['pollutant_id'].value_counts()


# In[41]:


b.loc[(b["pollutant_avg"]<=5),'Pollution Level'] = 'No'
b.loc[(b["pollutant_avg"]>5) & (b["pollutant_avg"]<=40),'Pollution Level'] = 'Moderate'
b.loc[(b["pollutant_avg"]>40) & (b["pollutant_avg"]<=80),'Pollution Level'] = 'Substantial'
b.loc[(b["pollutant_avg"]>80) & (b["pollutant_avg"]<=120),'Pollution Level'] = 'High'
b.loc[(b["pollutant_avg"]>120) & (b["pollutant_avg"]<=160),'Pollution Level'] = 'Worse'
b.loc[(b["pollutant_avg"]>160),'Pollution Level'] = 'Severe'


# In[42]:


h=b.sample(50)


# In[43]:


low = h[h["Pollution Level"] == 'Severe']["pollutant_avg"]
below_av = h[h["Pollution Level"] == 'Worse']["pollutant_avg"]
av = h[h["Pollution Level"] == 'High']["pollutant_avg"]
mod = h[h["Pollution Level"] == 'Substantial']["pollutant_avg"]
good = h[h["Pollution Level"] == 'Moderate']["pollutant_avg"]
minimal = h[h["Pollution Level"] == 'No']["pollutant_avg"]


# In[44]:


f_stats, p_value = sc.stats.f_oneway(low,below_av,av,mod,good,minimal,axis=0)


# In[45]:


print("F-Statistic={0}, P-value={1}".format(f_stats,p_value))


# In[46]:


pearson,p_value = sc.stats.pearsonr(h["pollutant_min"],h["pollutant_max"])


# In[47]:


print("Pearson Coefficient value={0}, P-value={1}".format(pearson,p_value))


# In[48]:


h.columns


# In[49]:


y = h['pollutant_avg']
x = h['pollutant_min']


# In[50]:


x = sm.add_constant(x)


# In[51]:


Result=sm.OLS(y,x).fit()


# In[52]:


Result.summary()


# In[53]:


h.corr()


# In[54]:


fig=plt.figure(figsize=(10,9))
sns.heatmap(h.corr(), cmap = 'viridis', annot=True,fmt='.2f',linewidths=2, cbar=False)
plt.title("Correlation Matrix (After Hypothesis Testing)")
plt.show() 


# In[ ]:




