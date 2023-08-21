import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task


# @task(retries=3, retry_delay_seconds=2)
# def read_data(filename: str) -> pd.DataFrame:
#     """Read data into DataFrame"""
#     df = pd.read_parquet(filename)

#     df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
#     df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

#     df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
#     df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

#     df = df[(df.duration >= 1) & (df.duration <= 60)]

#     categorical = ["PULocationID", "DOLocationID"]
#     df[categorical] = df[categorical].astype(str)

#     return df

@task(retries=3, retry_delay_seconds=2)
def read_split_data(filename: str, split_ratio=0.2) -> pd.DataFrame:
    """Read data into DataFrame"""
    data = pd.read_csv(filename)
    n = len(data)

    n_val = int(n * split_ratio)
    n_train = n - (n_val)

    idx = np.arange(n)
    shuffled_idx = np.random.permutation(n)
    

    val_idx = shuffled_idx[:n_val]
    train_idx = shuffled_idx[n_val:]
    
    return data.iloc[train_idx], data.iloc[val_idx]


# @task
# def add_features(
#     df_train: pd.DataFrame, df_val: pd.DataFrame
# ) -> tuple(
#     [
#         scipy.sparse._csr.csr_matrix,
#         scipy.sparse._csr.csr_matrix,
#         np.ndarray,
#         np.ndarray,
#         sklearn.feature_extraction.DictVectorizer,
#     ]
# ):
#     """Add features to the model"""
#     df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
#     df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

#     categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
#     numerical = ["trip_distance"]

#     dv = DictVectorizer()

#     train_dicts = df_train[categorical + numerical].to_dict(orient="records")
#     X_train = dv.fit_transform(train_dicts)

#     val_dicts = df_val[categorical + numerical].to_dict(orient="records")
#     X_val = dv.transform(val_dicts)

#     y_train = df_train["duration"].values
#     y_val = df_val["duration"].values
#     return X_train, X_val, y_train, y_val, dv

@task
def preprocess(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    
    categorical = ['cut', 'color', 'clarity']
    numerical = ['carat', 'depth', 'table', 'x', 'y', 'z']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    
    y_train = df_train["price"].values
    y_val = df_val["price"].values
    
    return X_train, X_val, y_train, y_val, dv






@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        # train = xgb.DMatrix(X_train, label=y_train)
        # valid = xgb.DMatrix(X_val, label=y_val)

        best_params =  {
        'eta': 0.1511729424139505,
        'max_depth': 10,
        'learning_rate': 0.011071197997506775,
        'n_estimators': 786,
        'min_child_weight': 0.7287981447110424,
        'gamma': 0.533419348964096,
        'subsample': 0.48587847813721813,
        'colsample_bytree': 0.7851620850095639,
        'reg_alpha': 0.7079483533181465,
        'reg_lambda': 0.040832236993174814,
        'random_state': 42
    }

        mlflow.log_params(best_params)
        
        xgb_reg = xgb.XGBRegressor(**best_params)
        xgb_reg.fit(X_train, y_train)
        y_pred = xgb_reg.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        
        # mlflow.log_params(best_params)

        # booster = xgb.train(
        #     params=best_params,
        #     dtrain=train,
        #     num_boost_round=100,
        #     evals=[(valid, "validation")],
        #     early_stopping_rounds=20,
        # )

        # y_pred = booster.predict(valid)
        # rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(xgb_reg, artifact_path="models_mlflow")
    return None


@flow
def main_flow(
    train_path: str = "./data/diamonds.csv",
    # val_path: str = "./data/green_tripdata_2021-02.parquet",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    TRACKING_SERVER_HOST = "ec2-34-227-160-154.compute-1.amazonaws.com" # fill in with the public DNS of the EC2 instance
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("xgb-diamond")

    # Load
    df_train, df_val = read_split_data(train_path)
    # df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = preprocess(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    main_flow()