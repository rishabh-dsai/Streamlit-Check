# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:46:23 2022

@author: asus
"""


import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

#%%

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_excel(uploaded_file)

else:
    st.warning('Please Upload File')


uploaded_model=st.file_uploader('Choose a model')
if uploaded_model is not None:
    model = joblib.load(uploaded_model)

else:
    st.warning('Please Upload Model')

#%%

X=pd.get_dummies(dataframe,prefix='State',columns=['State'])
X=X[['R&D Spend', 'Administration', 'Marketing Spend', 'State_California',
       'State_Florida', 'State_New York']]
y_preds=model.predict(X)

predictions=pd.DataFrame(y_preds,columns=['Predictions'])

dataframe['Predictions']=predictions
st.write(dataframe)

st.bar_chart(predictions[['Predictions']])
















