import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("loan-train.csv")

# Data Preprocessing
data = dataset.copy()

# Fill missing values
data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
data["Married"].fillna(data["Married"].mode()[0], inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0], inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0], inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0], inplace=True)
data["LoanAmount"].fillna(data["LoanAmount"].mean(), inplace=True)

# Drop irrelevant columns
data = data.drop(columns=["Loan_ID"])

# Encode categorical variables
Le_gender = LabelEncoder()
Le_married = LabelEncoder()
Le_dependents = LabelEncoder()
Le_self_employed = LabelEncoder()
Le_property_area = LabelEncoder()
Le_education = LabelEncoder()

# Fit on the training data and transform
data["Gender"] = Le_gender.fit_transform(data["Gender"])
data["Married"] = Le_married.fit_transform(data["Married"])
data["Dependents"] = Le_dependents.fit_transform(data["Dependents"])
data["Self_Employed"] = Le_self_employed.fit_transform(data["Self_Employed"])
data["Property_Area"] = Le_property_area.fit_transform(data["Property_Area"])
data["Education"] = Le_education.fit_transform(data["Education"])

# Splitting the dataset into features and target variable
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Predictions
y_pred = classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Loan Eligibility Prediction")

# User input
def user_input_features():
    gender = st.selectbox("Gender", ("Male", "Female"))
    married = st.selectbox("Married", ("Yes", "No"))
    dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.selectbox("Self Employed", ("Yes", "No"))
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", ("0", "1"))
    property_area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

    user_data = {
        "Gender": Le_gender.transform([gender])[0],
        "Married": Le_married.transform([married])[0],
        "Dependents": Le_dependents.transform([dependents])[0],
        "Education": Le_education.transform([education])[0],
        "Self_Employed": Le_self_employed.transform([self_employed])[0],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": int(credit_history),
        "Property_Area": Le_property_area.transform([property_area])[0],
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

# Input features from user
input_data = user_input_features()

# Display input data
st.write("User Input Parameters:")
st.write(input_data)

# Make predictions
prediction = classifier.predict(input_data)

# Output prediction result
st.subheader("Prediction:")
st.write("Eligible" if prediction == 1 else "Not Eligible")

# Display model accuracy
st.subheader("Model Accuracy:")
st.write(f"{accuracy * 100:.2f}%")
