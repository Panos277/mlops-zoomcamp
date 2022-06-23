#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import numpy as np
import argparse

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(args):
    
    year = args.year
    month = args.month
    df = read_data(f'./data/fhv_tripdata_{year:04d}-{month:02d}.parquet')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predictions'] = y_pred

    print("Mean duration : ",np.mean(y_pred))


if __name__ == "__main__":
    categorical = ['PUlocationID', 'DOlocationID']
    
    parser = argparse.ArgumentParser(description="run inference on nyc dataset")
    parser.add_argument("--year",type=int,default=2021)
    parser.add_argument("--month",type=int,default=4)
    
    args = parser.parse_args()
    main(args)

# df.to_parquet(
#     './data/output/fhv_tripdata_2021-02_predicted.parquet',
#     engine='pyarrow',
#     compression=None,
#     index=False
# )




