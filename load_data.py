import os
import pandas as pd
from datetime import datetime

def convert_time(time_str):
    return_time = datetime.strptime(time_str, '[Timestamp(\'%Y-%m-%d %H:%M:%S\')]')
    return return_time

import warnings
warnings.filterwarnings("ignore")
f = open('covid.txt').read()
count = 0

for line in f.split('\n'):
    tmp = line.split(',')
    pat_id = tmp[0]
    if pat_id == 'AA0HAI1_Illness_1':
        filename = 'AA0HAI1_1'
    elif pat_id == 'AA0HAI1_Illness_2':
        filename = 'AA0HAI1_2'
    elif pat_id == 'AA0HAI1_Illness_3':
        filename = 'AA0HAI1_3'
    else:
        filename = pat_id
    disease = tmp[1]
    symtom_date = convert_time(tmp[2])
    diagnosis_date = convert_time(tmp[3])
    recovery_date = convert_time(tmp[4])

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
    df1.to_csv('covid_df/%s.csv'%filename)