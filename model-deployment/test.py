import requests
import numpy as np

# features = {
#     "PULocationID": 10,
#     "DOLocationID": 50,
#     "trip_distance": 40
# }

raw_features = {
    "carat": 1.14,
    "cut": "Very Good",
    "color": "I",
    "clarity": "VS2",
    "depth": 60.0,
    "table": 45.0,
    "x": 6.8,
    "y": 8.75,
    "z": 4.06
    }

#input_array = np.array(list(raw_features.values())).reshape(1, -1)

url = 'http://localhost:9696/predict'
# response = requests.post(url, json=ride)
response = requests.post(url, json=raw_features)
print(response.json())