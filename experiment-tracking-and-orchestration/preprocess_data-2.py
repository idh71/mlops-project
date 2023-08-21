import os
import pickle
import click
import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)



def split_test_set(filename, split_ratio=0.2):
    
    data = pd.read_csv(filename)
    n = len(data)

    n_val = int(n * split_ratio)
    n_test = int(n * split_ratio)
    n_train = n - (n_val + n_test)

    np.random.seed(42)
    
    idx = np.arange(n)
    shuffled_idx = np.random.permutation(n)
    

    val_idx = shuffled_idx[:n_val]
    test_idx = shuffled_idx[n_val: n_val + n_test]
    train_idx = shuffled_idx[n_val + n_test:]
    
    return data.iloc[train_idx], data.iloc[val_idx], data.iloc[test_idx]

# def read_dataframe(filename: str):
#     df = pd.read_parquet(filename)

#     df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
#     df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
#     df = df[(df.duration >= 1) & (df.duration <= 60)]

#     categorical = ['PULocationID', 'DOLocationID']
#     df[categorical] = df[categorical].astype(str)

#     return df


def preprocess(df: pd.DataFrame):
    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['cut', 'color', 'clarity']
    numerical = ['carat', 'depth', 'table', 'x', 'y', 'z']
    dicts = df[categorical + numerical].to_dict(orient='records')
    # if fit_dv:
    #     X = dv.fit_transform(dicts)
    # else:
    #     X = dv.transform(dicts)
    return dicts


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):
    # Load parquet files
    # df_train = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2022-01.parquet")
    # )
    # df_val = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2022-02.parquet")
    # )
    # df_test = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet")
    # )
    df_train, df_val, df_test = split_test_set(raw_data_path, split_ratio=0.2)

    # Extract the target
    target = 'price'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    # dv = DictVectorizer()
    # X_train, dv = preprocess(df_train, dv, fit_dv=True)
    # X_val, _ = preprocess(df_val, dv, fit_dv=False)
    # X_test, _ = preprocess(df_test, dv, fit_dv=False)
    dict_train = preprocess(df_train)
    dict_val = preprocess(df_val)
    dict_test= preprocess(df_test)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    # dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    # dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    # dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    # dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
    # dump_pickle((dv), os.path.join(dest_path, "dv.pkl"))

    # dump_pickle((df_train, y_train), os.path.join(dest_path, "df_train.pkl"))
    # dump_pickle((df_val, y_val), os.path.join(dest_path, "df_val.pkl"))
    # dump_pickle((df_test, y_test), os.path.join(dest_path, "df_test.pkl"))
    # dump_pickle((dv), os.path.join(dest_path, "dv.pkl"))

    dump_pickle((dict_train, y_train), os.path.join(dest_path, "dict_train.pkl"))
    dump_pickle((dict_val, y_val), os.path.join(dest_path, "dict_val.pkl"))
    dump_pickle((dict_test, y_test), os.path.join(dest_path, "dict_test.pkl"))
    # dump_pickle((dv), os.path.join(dest_path, "dv.pkl"))


if __name__ == '__main__':
    run_data_prep()