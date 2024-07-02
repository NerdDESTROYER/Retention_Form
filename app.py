import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
import requests

# Define the form fields as per the HTML file
with open('employee1.pkl', 'rb') as model_file:
    model = pkl.load(model_file)
    
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pkl.load(scaler_file)

# Define mappings for categorical fields
BusinessTravel = {"Travel_Rarely": 2, "Travel_Frequently": 1, "Non-Travel": 0}
Department = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}
EducationField = {
    "Technical Degree": 5,
    "Other": 4,
    "Medical": 3,
    "Marketing": 2,
    "Life Sciences": 1,
    "Human Resources": 0
}
JobRole = {
    "Sales Representative": 8,
    "Sales Executive": 7,
    "Research Scientist": 6,
    "Research Director": 5,
    "Manufacturing Director": 4,
    "Manager": 3,
    "Laboratory Technician": 2,
    "Human Resources": 1,
    "Healthcare Representative": 0
}
MaritalStatus = {"Single": 2, "Married": 1, "Divorced": 0}
OverTime = {"Yes": 1, "No": 0}

def model_prediction(form_data):
    # Collect data from form
       
    # Convert data to DataFrame
    data = pd.DataFrame([form_data])

    # Define columns for data1 and data2
    data1_columns = ['Age', 'DailyRate', 'Education', 'HourlyRate', 'JobLevel','MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike','StockOptionLevel', 'TotalWorkingYears', 'WorkLifeBalance','YearsSinceLastPromotion', 'YearsWithCurrManager']

    data2_columns = ['BusinessTravel', 'Department', 'EducationField','JobRole', 'MaritalStatus', 'OverTime']

# Split the data into data1 and data2
    data1 = data[data1_columns]
    data2 = data[data2_columns]
    
    # Convert data types as necessary
    data1 = data1.astype({
        'Age': int,
        'DailyRate': int,
        'Education': int,
        'HourlyRate': int,
        'JobLevel': int,                
        'MonthlyRate': int,
        'NumCompaniesWorked': int,        
        'PercentSalaryHike': int,
        'StockOptionLevel': int,
        'TotalWorkingYears': int,
        'WorkLifeBalance': int,
        'YearsSinceLastPromotion': int,
        'YearsWithCurrManager': int
    })
    data2 = data2.astype({
        'BusinessTravel': int,
        'Department': int,
        'EducationField': int, 
        'JobRole': int,
        'MaritalStatus': int,
        'OverTime': int     
    })

    data_new=scaler.transform(data1)  
    merged_data = pd.concat([data1, data2], axis=1)
 
    prediction = model.predict(merged_data)
    print(prediction)

    if(prediction[0]==1):
        return f'Prediction : Employee will stay'
    else:
        return f'Prediction : Employee will quit'   
    
def main():
    st.title("EMPLOYEE RETENTION FORM")

    Age = st.text_input("Age")
    BusinessTravel_input = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=0)
    DailyRate = st.text_input("Daily Rate")
    Department_input = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], index=0)
    Education = st.text_input("Education")
    EducationField_input = st.selectbox("Education Field", ["Technical Degree", "Other", "Medical", "Marketing", "Life Sciences", "Human Resources"], index=0)
    HourlyRate = st.text_input("Hourly Rate")
    JobLevel = st.text_input("Job Level")
    JobRole_input = st.selectbox("Job Role", ["Sales Representative", "Sales Executive", "Research Scientist", "Research Director", "Manufacturing Director", "Manager", "Laboratory Technician", "Human Resources", "Healthcare Representative"], index=0)
    MaritalStatus_input = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
    MonthlyRate = st.text_input("Monthly Rate")
    NumCompaniesWorked = st.text_input("Number of Companies Worked")
    OverTime_input = st.selectbox("Over Time", ["Yes", "No"], index=0)
    PercentSalaryHike = st.text_input("Percent Salary Hike")
    StockOptionLevel = st.text_input("Stock Option Level")
    TotalWorkingYears = st.text_input("Total Working Years")
    WorkLifeBalance = st.text_input("Work Life Balance")
    YearsSinceLastPromotion = st.text_input("Years Since Last Promotion")
    YearsWithCurrManager = st.text_input("Years With Current Manager")

    diagnosis = ''

    if st.button("Predict"):

        required_fields = {
            "Age": Age,
            "DailyRate": DailyRate,
            "Education": Education,
            "HourlyRate": HourlyRate,
            "JobLevel": JobLevel,
            "MonthlyRate": MonthlyRate,
            "NumCompaniesWorked": NumCompaniesWorked,
            "PercentSalaryHike": PercentSalaryHike,
            "StockOptionLevel": StockOptionLevel,
            "TotalWorkingYears": TotalWorkingYears,
            "WorkLifeBalance": WorkLifeBalance,
            "YearsSinceLastPromotion": YearsSinceLastPromotion,
            "YearsWithCurrManager": YearsWithCurrManager
        }
        
        # Check for empty fields
        if any(value.strip() == "" for value in required_fields.values()):
            st.write("Please fill out all required fields.")
            return
        
        form_data = {
            "Age": Age,
            "BusinessTravel": BusinessTravel[BusinessTravel_input],
            "DailyRate": DailyRate,
            "Department": Department[Department_input],
            "Education": Education,
            "EducationField": EducationField[EducationField_input],
            "HourlyRate": HourlyRate,
            "JobLevel": JobLevel,
            "JobRole": JobRole[JobRole_input],
            "MaritalStatus": MaritalStatus[MaritalStatus_input],
            "MonthlyRate": MonthlyRate,
            "NumCompaniesWorked": NumCompaniesWorked,
            "OverTime": OverTime[OverTime_input],
            "PercentSalaryHike": PercentSalaryHike,
            "StockOptionLevel": StockOptionLevel,
            "TotalWorkingYears": TotalWorkingYears,
            "WorkLifeBalance": WorkLifeBalance,
            "YearsSinceLastPromotion": YearsSinceLastPromotion,
            "YearsWithCurrManager": YearsWithCurrManager
        }
        
        diagnosis = model_prediction(form_data)

    st.success(diagnosis)


if __name__ == '__main__':
    main()
