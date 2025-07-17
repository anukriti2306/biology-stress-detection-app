import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

# Load model
model_file = r'model/final_model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Labels
stress_labels = {
    1: "Low/Normal (No stress)",
    2: "Medium low (Mild stress)",
    3: "Medium (Moderate stress)",
    4: "Medium (Significant stress)",
    5: "High (Concerning)"
}

# Features used in model
most_important_features = ['heart_rate', 'sleeping_hours', 'snoring_rate', 'body_temperature']

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for your Streamlit frontend
CORS(app, resources={
    r"/predict": {"origins": "https://stress-detector-a23.streamlit.app"}
})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        input_data = {key: float(data[key]) for key in most_important_features}
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input type: {e}'}), 400

    X_input = dv.transform([input_data])
    model_output = int(model.predict(X_input)[0])  # 1 to 5
    description = stress_labels.get(model_output, 'Unknown')

    return jsonify({
        'stress_level': model_output,
        'description': description
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
