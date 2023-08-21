import os
import pickle
import pandas as pd

import mlflow
from flask import Flask, request, jsonify


# def load_pickle(filename):
#     with open(filename, "rb") as f_in:
#         return pickle.load(f_in)



EXPERIMENT_NUMBER = '27'
RUN_ID = '9555504150e34843985d62dbfa88cd13'


logged_model = f's3://mlflow-artifacts-remote-41/{EXPERIMENT_NUMBER}/{RUN_ID}/artifacts/mlflow_models'
# logged_model = f'runs:/{RUN_ID}/model'

model = mlflow.pyfunc.load_model(logged_model)

def predict(raw_features):
    # features = dv.transform(raw_features)
    preds = model.predict(raw_features)
    return float(preds[0])


app = Flask('price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
   
    raw_features = request.get_json()

    # dv = load_pickle('/Users/isaachurwitz/my-project/02-experiment-tracking/models/preprocessor.b')

    
    pred = predict(raw_features)

  

    result = {
        'predicted_price': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)