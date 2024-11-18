import pickle
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained XGBClassifier model
with open('water_quality_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        ph = float(request.form['ph'])
        hardness = float(request.form['Hardness'])
        solids = float(request.form['Solids'])
        chloramines = float(request.form['Chloramines'])
        sulfate = float(request.form['Sulfate'])
        conductivity = float(request.form['Conductivity'])
        organic_carbon = float(request.form['Organic_carbon'])
        trihalomethanes = float(request.form['Trihalomethanes'])
        turbidity = float(request.form['Turbidity'])

        # Prepare the input array for prediction
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                                conductivity, organic_carbon, trihalomethanes, turbidity]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Map the prediction (0 or 1) to a label for clarity
        result = "Potable" if prediction[0] == 1 else "Not Potable"
        
        # Return the prediction result
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
