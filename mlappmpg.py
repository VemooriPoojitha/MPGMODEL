#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[35]:





# In[36]:


mpgdf = pd.read_csv("Auto MPG Reg.csv")


# In[37]:


mpgdf.horsepower = pd.to_numeric(mpgdf.horsepower, errors="coerce")


# In[38]:


mpgdf.horsepower = mpgdf.horsepower.fillna(mpgdf.horsepower.median())


# In[39]:


# Split data

y =mpgdf.mpg
X=mpgdf.drop(['carname','mpg'],axis=1)


# In[40]:


# Define multiple models as a dictionary

models = {'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(),'Random Forest':RandomForestRegressor(),
          'GradientBoosting':GradientBoostingRegressor()}


# In[41]:


# Sidebar for model selection

selected_model = st.sidebar.selectbox("Select a ML Model",list(models.keys()))


# In[42]:


# ML model selection parameters

if selected_model == 'Linear Regression':
    model = LinearRegression()
elif selected_model == 'Decision Tree':
    max_depth = st.sidebar.slider("max_depth",8,16,2)
    model = DecisionTreeRegressor(max_depth=max_depth)
elif selected_model == 'Random Forest':
    n_estimators = st.sidebar.slider("Num of Trees",100,500,50)
    model = RandomForestRegressor(n_estimators=n_estimators)
elif selected_model == 'GradientBoosting':
    n_estimators = st.sidebar.slider("Num of Trees",100,500,50)
    model = GradientBoostingRegressor(n_estimators=n_estimators)
    


# In[43]:


# Train the model

model.fit(X,y)


# In[44]:


# Define application page parameters

st.title("Predict Mileage per Gallon")
st.markdown("Model to ptredict Mileage of Car")
st.header("Car Features")

col1, col2, col3, col4 = st.columns(4)
with col1:
    cylinders = st.slider("Cylinders",2,8,1)
    displacement = st.slider("displacement",50,500,10)
with col2:
    horsepower = st.slider("HorsePower",50,500,10)
    weight = st.slider("Weight",1500,6000,250)
with col3:
    acceleration = st.slider("Accel",8,25,1)
    modelyear = st.slider("year",70,85,1)
with col4:
    origin = st.slider("origin",1,3,1)


# In[45]:


# Model Predictions
rsquare=model.score(X,y)
y_pred = model.predict(np.array([[cylinders, displacement,horsepower, weight, acceleration, modelyear, origin]]))


# In[46]:


# Display Results

st.header("ML Model Results")
st.write(f"Selected Model : {selected_model}")
st.write(f"RSquare : {rsquare}")
st.write(f"Predicted : {y_pred}")


# In[ ]:




