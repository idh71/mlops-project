import os
import pickle
import click
import mlflow
import xgboost as xgb

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
HPO_EXPERIMENT_NAME = "xgb-diamond-price"
# HPO_EXPERIMENT_NAME = "xgb-reg-hyperopt"
EXPERIMENT_NAME = "xgb-diamond-price-best-models"
#XGB_PARAMS = ['max_depth', 'n_estimators', 'random_state'] 
XGB_PARAMS = ['max_depth', 'n_estimators', 'random_state'] 

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
TRACKING_SERVER_HOST = "ec2-34-227-160-154.compute-1.amazonaws.com" # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.xgboost.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    # me experimenting
    dv = load_pickle(os.path.join(data_path, "dv.pkl"))

    with mlflow.start_run():
        for param in XGB_PARAMS:
            params[param] = int(params[param])

        xgb_reg = xgb.XGBRegressor(**params)
        xgb_reg.fit(X_train, y_train)


        

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, xgb_reg.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, xgb_reg.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_artifact(os.path.join(data_path, "dv.pkl"),  'artifacts')
        

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="xgb-reg-best-model")


if __name__ == '__main__':
    run_register_model()