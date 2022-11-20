from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model import ml_predict

app = Flask(__name__)
api = Api(app)

class ml_inference(Resource):
    def __init__(self):
        super().__init__()
        # choose default (average) values for optional entries
        self.ml_inputs = {
            "Pregnancies": 2,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 88,
            "BMI": 29,
            "DiabetesPedigreeFunction": 0.236,
            "Age": 35
        }
        self.mandatory_inputs = ["Glucose", "BMI", "Age"]
        self.optional_inputs = ["Pregnancies", "BloodPressure", "SkinThickness", \
            "Insulin", "DiabetesPedigreeFunction"]

    def post(self):
        json_data = request.get_json(force=True)
        for feature in self.mandatory_inputs:
            if feature not in json_data:
                return jsonify(response = f"Mandatory input ({feature}) is missing")
            self.ml_inputs[feature] = json_data[feature]
        
        for feature in self.optional_inputs:
            if feature in json_data:
                self.ml_inputs[feature] = json_data[feature]

        return jsonify(diabetes_probability = ml_predict(self.ml_inputs))

api.add_resource(ml_inference, '/diabetes_prediction')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444 ,debug=True)