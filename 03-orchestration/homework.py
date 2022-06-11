import pandas as pd
import pickle
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task
from prefect.engine import get_run_logger

from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f'The mean duration of training is {mean_duration}')
    else:
        logger.info(f'The mean duration of validation is {mean_duration}')
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    
    date_train = date - relativedelta(months=2)
    date_val = date - relativedelta(months=1)
    
    train_path = f'./data/fhv_tripdata_{date_train.year}-{ date_train.isoformat().split("-")[1] }.parquet'
    val_path = f'./data/fhv_tripdata_{date_train.year}-{ date_train.isoformat().split("-")[1] }.parquet'
    
    return train_path,val_path


@flow
def main(date_data="2021-03-15"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    if date_data==None:
        train_path, val_path = get_paths(date.today()).result()
    else:
        train_path, val_path = get_paths(datetime.strptime(date_data,"%Y-%m-%d")).result()        
    print(train_path,val_path)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    
    model_path = f'artifacts_local/lr-{date_data}.bin'
    preprocessor_path = f'artifacts_local/dv-{date_data}.b'
    dump_pickle(dv,preprocessor_path)
    dump_pickle(lr,model_path)
    
    mlflow.log_artifact(model_path,artifact_path="models")
    mlflow.log_artifact(preprocessor_path,artifact_path="preprocessors")
    
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    name="cron-schedule-deployment-training",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    tags=["ml"],
    flow_runner=SubprocessFlowRunner()
)