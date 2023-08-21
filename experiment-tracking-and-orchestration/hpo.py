import os
import pickle
import click
import mlflow
import optuna
import xgboost as xgb
import boto3

from optuna.samplers import TPESampler
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#mlflow.set_tracking_uri("http://127.0.0.1:5000")

TRACKING_SERVER_HOST = "ec2-34-227-160-154.compute-1.amazonaws.com" # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_experiment("xgb-diamond-price-final")

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed diamond data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)

def run_optimization(data_path: str, num_trials: int):
    mlflow.xgboost.autolog(disable=True)

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        params = {
        'eta': trial.suggest_float('eta', 0.1, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 2.5),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': 42

    }
        with mlflow.start_run():
            
            mlflow.log_params(params)
            
            xgb_reg = xgb.XGBRegressor(**params)
            xgb_reg.fit(X_train, y_train)
            y_pred = xgb_reg.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return rmse
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

    print('Best parameters', study.best_params)
    print('Best value', study.best_value)

if __name__ == '__main__':
    run_optimization()


     