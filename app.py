import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Simple Linear Regression with Streamlit")

data = pd.read_csv("study.csv")

x = data[['Hours_Studied']]
y = data['Marks_Scored']

model = LinearRegression()
model.fit(x, y)

hours = st.number_input("Enter Hours Studied:", 1.0, 10.0, step=0.5)
prediction = model.predict([[hours]])
st.write("Predicted Marks:", prediction[0])


st.title("House Price Prediction")

data = pd.read_csv("house_data.csv")


x = data[['Area_in_Sqft']]
y = data['Price_in_Thousands']

model = LinearRegression()
model.fit(x, y)

area = st.number_input("Enter Area in Sqft:", 500.0, 5000.0, step=100.0)
prediction = model.predict([[area]])
st.write("Predicted Price (in Thousands):", prediction[0])

st.title ("Salary Prediction Based on Experience and Age")

data = pd.read_csv("salary_data.csv")

x = data[['Experience_in_Years', 'Age']]
y = data['Salary_in_Thousands']

model = LinearRegression()
model.fit(x, y)

experience = st.number_input("Enter Experience in Years:", 0.0, 20.0, step=1.0)
age = st.number_input("Enter Age:", 18.0, 60.0, step=1.0)

prediction = model.predict([[experience, age]])
st.write("Predicted Salary (in Thousands):", prediction[0])


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

st.title("Student Result Prediction App")

data = pd.read_csv("student_data.csv")
st.write(data)

x = data[['Marks', 'Attendance']]
y = data['Result'].map({'Fail': 0, 'Pass': 1})

model = DecisionTreeClassifier()
model.fit(x, y)

marks = st.number_input("Enter Marks:", 0.0, 100.0, step=1.0)
attendance = st.number_input("Enter Attendance (%):", 0.0, 100.0, step=1.0)

prediction = model.predict([[marks, attendance]])[0]
result = "Pass" if prediction == 1 else "Fail"

st.write("Predicted Result:", result)