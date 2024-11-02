from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('gradient_boosting_model.pkl')

app = Flask(__name__)

#print("Model expected input features:", model.n_features_in_)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect demographic and physiological input data
        age = float(request.form['age'])
        gender = request.form['gender']
        heart_rate = float(request.form['heart_rate'])
        blood_pressure = float(request.form['blood_pressure'])
        respiratory_rate = float(request.form['respiratory_rate'])
        body_temperature = float(request.form['body_temperature'])
        comorbidity_count = int(request.form['comorbidity_count'])
        income_level = request.form['income_level']

        # Process categorical variables
        gender_male = 1 if gender == 'male' else 0
        
        income_low, income_medium, income_high = 0, 0, 0
        if income_level == 'low':
            income_low = 1
        elif income_level == 'medium':
            income_medium = 1
        elif income_level == 'high':
            income_high = 1

        # Define features array with collected values
        features = [
            age,
            gender_male,
            heart_rate,
            blood_pressure,
            respiratory_rate,
            body_temperature,
            comorbidity_count,
            income_low,
            income_medium,
            income_high,
        ]

        # Add placeholders to reach 31 features
        features += [0] * (31 - len(features))  # Adjust length to match model's input size

        # Make prediction
        prediction = model.predict([features])[0]
        
        return f'The predicted hospital stay is {prediction:.2f} days.'

    except ValueError as e:
        return f'Error: Invalid input - {str(e)}'
    except Exception as e:
        return f'An error occurred: {str(e)}'
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    
    


