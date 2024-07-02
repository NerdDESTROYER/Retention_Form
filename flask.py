from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('employee1.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    form_data = request.form.to_dict()
    
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
    
    return f'Prediction: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)