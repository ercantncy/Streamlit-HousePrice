import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set()
import plotly
import matplotlib.pyplot as plt

import random as rnd

from sklearn.preprocessing import StandardScaler
import scipy
import streamlit as st 
import pickle

st.write(" # House Price Prediction #")
df = pd.read_csv('final.csv') 
liste=["HighQualSF","GarageCars","CentralAir",
	"Exterior1st","BsmtQual","Total_Bathrooms",
	"OverallQual","TotalBsmtSF","Total_Home_Quality"]


s1 = st.sidebar.slider('Total Square Feet',min_value=0, max_value=5000)

cycle_2 = [0, 1 , 2,  3,  4]
s2 = st.sidebar.selectbox("Garage Cars",options=cycle_2)

cycle_3 = ['Y', 'N']
s3 = st.sidebar.selectbox("CentralAir",options=cycle_3)

cycle_4 = ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock',
	'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood',
	'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']
s4 = st.sidebar.selectbox("Exterior covering on house",options=cycle_4)

cycle_5 = ['Gd', 'TA', 'Ex', 'None', 'Fa']
s5 = st.sidebar.selectbox("Basement Quality",options=cycle_5)

cycle_6 = [0, 1 , 2,  3,  4, 5, 6]
s6 = st.sidebar.selectbox("Total Bathrooms",options=cycle_6)

s7=st.sidebar.slider("Overall Quality",min_value=0, max_value=10)

s8=st.sidebar.slider("Total square feet of basement area",min_value=0, max_value=6500)

s9=st.sidebar.slider("Total Home Quality",min_value=1, max_value=20)


dict = {'HighQualSF':[s1],'GarageCars':[s2],'CentralAir':[s3],'Exterior1st':[s4],'BsmtQual':[s5],'Total_Bathrooms':[s6],'OverallQual':[s7],
        'TotalBsmtSF':[s8],'Total_Home_Quality':[s9]}
dictf = pd.DataFrame(dict)
df = df.append(dictf,ignore_index= True) 
for i in df.columns:
    if df[i].dtypes in ["object"]:
        df[i].fillna(df[i].mode(),inplace = True)
    else:
        df[i].fillna(df[i].mean(),inplace = True)
        



df.drop('Unnamed: 0', axis= 1, inplace= True)

df2 = pd.get_dummies(df)
scaler = StandardScaler()
scaler.fit(df2)
df3 = pd.DataFrame(scaler.transform(df2),index = df2.index,columns = df2.columns)


newdata=pd.DataFrame(df3.iloc[[-1]])


with open('finalized_model.sav' , 'rb') as f:
    lr = pickle.load(f)


if st.sidebar.button('Show House Price'):
    ypred = lr.predict(newdata)
    st.title('Predicted Price : ')
    st.title(ypred[0])



