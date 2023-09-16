#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("F:/Project/quora-question-pairs/train.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar')


# In[9]:


qid=pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Number of unique questions',np.unique(qid).shape[0])
x=qid.value_counts()>1
print('Number of questions getting repeated',x[x].shape[0])


# In[10]:


plt.hist(qid.value_counts().values,bins=160)
plt.yscale('log') 
plt.show()


# In[11]:


ques_df = df[['question1','question2']] 
ques_df.head()


# In[12]:


df.isna().sum()


# In[13]:


ques_df.shape


# In[14]:


a=ques_df.dropna()


# In[15]:


ques_df=a.drop(1)


# In[16]:


ques_df.shape


# In[17]:


new_df = df.sample(30000)


# In[18]:


new_df.isnull().sum()


# In[19]:


new_df.duplicated().sum()


# In[20]:


ques_df = new_df[['question1','question2']] 
ques_df.head()


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
ques_df.dropna(subset=['question1', 'question2'], inplace=True)
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)


# In[22]:


temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


# In[23]:


temp_df


# In[24]:


temp_df['is_duplicate'] = new_df['is_duplicate']


# In[25]:


temp_df.head()


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(temp_df.iloc[:,0:-1].values,temp_df.iloc[:,-1].values,test_size=0.2,random_state=1)


# In[27]:


df.head()


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[29]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.svm import SVC
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

