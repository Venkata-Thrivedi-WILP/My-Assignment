from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Initialize Swagger API documentation
api = Api(app, version='1.0', title='Iris Model API', description='API to predict Iris flower species')

# Define the input model for Swagger
predict_model = api.model('Prediction', {
    'input': fields.List(fields.Float, required=True, description='List of feature values (sepal_length, sepal_width, petal_length, petal_width)')
})

# Load the model from the disk
model = joblib.load("models/best_rf_model.pkl")

# Define prediction endpoint using Swagger and POST method
@api.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)  # Swagger expects the model structure
    def post(self):
        try:
            # Get data from POST request
            data = request.get_json()

            # Ensure the input data is in the right shape
            input_data = np.array(data["input"]).reshape(1, -1)

            # Make predictions
            prediction = model.predict(input_data)

            # Return prediction as JSON
            return jsonify({"prediction": int(prediction[0])})

        except Exception as e:
            return jsonify({"error": str(e)})


# Define a simple 'health' endpoint to test if the API is running
@api.route('/health')
class HealthCheck(Resource):
    def get(self):
        return {'status': 'OK'}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
