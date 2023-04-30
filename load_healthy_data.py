import os
import pandas as pd
from datetime import datetime

def convert_time(time_str):
    return_time = datetime.strptime(time_str, '[Timestamp(\'%Y-%m-%d %H:%M:%S\')]')
    return return_time

import warnings
warnings.filterwarnings("ignore")
f = open('healthy_ID.txt').read()
count = 0

for filename in f.split('\n'):
    if filename not in ['A3ADWUT','AA0HAI1']:#patients with invalid data
        df_hr = pd.read_csv('COVID-19-Wearables/%s_hr.csv'%filename)
        df_steps = pd.read_csv('COVID-19-Wearables/%s_steps.csv'%filename)
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)

        df_steps = df_steps.set_index('datetime')
        df_steps.index.name = None
        df_steps.index = pd.to_datetime(df_steps.index)

        df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
        #Hyparameter - resample resolution, and imputation method
        df1 = df1.resample('1min').mean()
        df1 = df1.ffill(axis=0)
        df1.to_csv('healthy_df/%s.csv'%filename)