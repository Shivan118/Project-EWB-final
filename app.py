import streamlit as st
# import preprocessor,helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model_final.pkl','rb'))
encoder = pickle.load(open('target_encoder.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))

st.title("Insurance Premium Prediction")
age = st.text_input('Enter Age', 18)
age = int(age)

sex = st.selectbox(
    'Please select gender',
    ('male', 'female'))
# gender = encoder.transform(np.array([sex]))

bmi = st.text_input('Enter BMI', 18)
bmi = float(bmi)

children = st.selectbox(
    'Please select number of children ',
    (0,1,2,3,4,5))
children = int(children)


smoker = st.selectbox(
    'Please select smoker category ',
    ("yes","no"))
# smoker = encoder.transform(smoker)

region = st.selectbox(
    'Please select region ',
    ("southwest", "southeast", "northeast", "northwest"))


l = {}
l['age'] = age
l['sex'] = sex
l['bmi'] = bmi
l['children'] = children
l['smoker'] = smoker
l['region'] = region

df = pd.DataFrame(l, index=[0])

df['region'] = encoder.transform(df['region'])
df['sex'] = df['sex'].map({'male':1, 'female':0})
df['smoker'] = df['smoker'].map({'yes':1, 'no':0})

df = transformer.transform(df)
# dtrain = xg.DMatrix(df)
y_pred = model.predict(df)
# st.write(age, gender, bmi, children, smoker, region)

if st.button("Show Result"):
    # col1,col2, col3,col4 = st.columns(4)
    st.header(f"{round(y_pred[0],2)} INR")
