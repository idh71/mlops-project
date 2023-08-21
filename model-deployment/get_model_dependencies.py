import mlflow.pyfunc

EXPERIMENT_NUMBER = '27' # Replace with your experiment number
RUN_ID = '9555504150e34843985d62dbfa88cd13'  # Replace with your run ID
model_uri = f's3://mlflow-artifacts-remote-41/{EXPERIMENT_NUMBER}/{RUN_ID}/'

# Fetch model dependencies
model_dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)


with open('model_dependencies.txt', 'w') as file:
    for dep in model_dependencies:
        file.write(dep + '\n')