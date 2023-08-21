import os
import pickle

import pandas as pd
import mlflow
from flask import Flask, request, jsonify


RUN_ID = os.getenv('RUN_ID')
RUN_ID = '37ab0d5aa3ac4de6817e8e0439e91584'

#logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
logged_model = f's3://mlflow-artifacts-remote-41/18/{RUN_ID}/artifacts/model'
# logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


# def prepare_features(ride):
#     features = {}
#     features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
#     features['trip_distance'] = ride['trip_distance']
#     return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


raw_features = {
    "carat": 1.14,
    "cut": "Good",
    "color": "I",
    "clarity": "SI2",
    "depth": 60.0,
    "table": 65.0,
    "x": 6.8,
    "y": 6.75,
    "z": 4.06
    }

features = pd.DataFrame(raw_features, index=raw_features.items())

predict(features)


# app = Flask('price-prediction')


# @app.route('/predict', methods=['POST'])
# def predict_endpoint():
#     # ride = request.get_json()
#     features = request.get_json()

#     # features = prepare_features(ride)
#     pred = predict(features)

#     # result = {
#     #     'duration': pred,
#     #     'model_version': RUN_ID
#     # }

#     result = {
#         'predicted_price': pred,
#         'model_version': RUN_ID
#     }

#     return jsonify(result)

# # @app.route('/predict', methods=['POST'])
# # def predict_endpoint():
# #     return jsonify({"message": "Success"})


# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=9696)