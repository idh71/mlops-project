# Diamond Price Prediction Project

![Diamonds](diamonds.jpg)

## Overview

This repository contains a data science and machine learning project focused on predicting the price of diamonds based on various attributes. The project utilizes a dataset sourced from Kaggle, containing information about diamond attributes such as carat, cut, color, clarity, dimensions, and more.

## Dataset

The dataset used for this project can be found on Kaggle: [Diamond Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds)

The dataset includes the following attributes:
- `price`: Price of the diamond in US dollars (\$326--\$18,823)
- `carat`: Weight of the diamond (0.2--5.01)
- `cut`: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- `color`: Diamond colour, from J (worst) to D (best)
- `clarity`: Measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- `x`: Length in mm (0--10.74)
- `y`: Width in mm (0--58.9)
- `z`: Depth in mm (0--31.8)
- `depth`: Total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- `table`: Width of top of diamond relative to the widest point (43--95)

## Goal

The main objective of this project is to build a machine learning model that can predict the price of diamonds accurately based on their attributes. To achieve this goal, we've followed these steps:

1. Data Preprocessing: Cleaning, transforming, and exploring the dataset to understand its structure and characteristics.
2. Feature Selection: Identifying relevant features that have a significant impact on diamond prices.
3. Model Building: Creating and training machine learning models using various regression techniques.
4. Model Evaluation: Assessing model performance using appropriate evaluation metrics.
5. Deployment: Deploying the best-performing model for real-world diamond price predictions.

## Tools Used

- **Jupyter Notebook**: Used for exploratory data analysis, feature engineering, and building machine learning models interactively.
- **MLflow**: Managed experiment tracking, model comparison, and performance management.
- **Prefect**: Orchestrated data preprocessing, model training, and evaluation pipelines.
- **Evidently**: Monitored data quality and model performance over time.
- **Grafana**: Created interactive dashboards for visualizing project metrics and insights.
- **XGBoost**: Employed for building regression models to predict diamond prices.

## Initial Environment Setup

1. Create conda environemnt `conda create --n diaond_project python=3.10`
2. Activate the virtual environment `conda activate diaond_project`
3. Install the required libraries: `pip install -r requirements.txt`

This environment can be used to run the diamond_price_prediction.ipnb notebook and the files in the experiment-tracking-and orchestration folders.  Other sections of the repository can be run using a pipenv environment created from the provided pipfile.

## Experiment Tracking And Workflow Orchestration

1. From the main project folder `cd experiment-tracking-and-orchestration`
2. `conda activate diaond_project`

I have provided files for preprocessing the raw data (preprocess_data.py), tuning the hyperparameters of the xgb regression model (hpo.py), registering the best version of the model to the mlflow model registry (register_model.py) and running a workflow orchrestration with prefect (orchestration.py). I used a remote mlflow server on an ec2 instance for experiment tracking tracking and model registry and prefect cloud for workflow orchestration.

## Model Deployment

1. From the main project folder `cd model-deployment`
2. `pipenv install`
3.  Activate the pipenv environement `pipenv shell`
4. Build the docker container from Dockerfile `docker build -t diamond-price-prediction-service:v1 .`
5. Start the docker container `docker run -it --rm -p 9696:9696  diamond-price-prediction-service:v1`

In another terminal (using the pipenv shell) you can test the prediction service by running `python test.py`.

## Data Monitoring

1. From the main project folder `cd diamond_data_monitoring`
2. `conda create --name monitoring_env python=3.11`
3. `conda activate monitoring_env`
4. `pip install -r requirements.txt`
5. `docker-compose up --build`
6. In another terminal tab run `python evidently_metrics_calculation.py`.
7. Open a browser to localhost:3000 to view the data metrics in grafana.  













