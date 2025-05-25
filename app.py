import streamlit as st
import pandas as pd


st.title("Diabetes Prediction App")

data = pd.read_csv("diabetes_data.csv")
st.write(data)

x = data[['Age', 'Glucose_Level']]
y = data['Diabetes'].map({'No': 0, 'Yes': 1})




