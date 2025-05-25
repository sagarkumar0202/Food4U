import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Diabetes Prediction App")

data = pd.read_csv("diabetes_data.csv")
st.write(data)

x = data[['Age', 'Glucose_Level']]
y = data['Diabetes'].map({'No': 0, 'Yes': 1})

model = LogisticRegression()
model.fit(x, y)

age = st.number_input("Enter Age:", 18.0, 100.0, step=1.0)
glucose = st.number_input("Enter Glucose Level:", 50.0, 200.0, step=1.0)

prediction = model.predict([[age, glucose]])[0]
result = "Yes" if prediction == 1 else "No"

st.write("Diabetes Prediction:", result)

