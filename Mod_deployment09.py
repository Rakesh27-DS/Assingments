#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import pickle 
import streamlit as st


# In[3]:


model=pickle.load(open('log14.pkl','rb'))


# In[6]:


model


# In[7]:


st.title('Model Deployment using streamlit')


# In[8]:


def user_defined_values():
    PassengerId=st.sidebar.number_input('PassengerId')
    Pclass=st.sidebar.number_input('Pclass')
    Sex=st.sidebar.selectbox('Sex,M=1,F=0',[0,1])
    Age=st.sidebar.number_input('Age')
    SibSp=st.sidebar.number_input('SibSp')
    Parch=st.sidebar.number_input('Parch')
    Fare=st.sidebar.number_input('Fare')
    Cabin=st.sidebar.number_input('Cabin')
    Embarked=st.sidebar.number_input('Embarked')
    data={'PassengerId':PassengerId,'Pclass':Pclass,'Sex':Sex,'Age':Age,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Cabin':Cabin,'Embarked':Embarked}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_defined_values()
st.subheader('User_Input_Variables')
st.write(df)
pred=model.predict(df)
pred_proba=model.predict_proba(df)
st.write('Yes' if pred_proba[0][1]>0.5 else 'No')
st.subheader('Pred_Proba')
st.write(pred_proba)


# In[ ]:




