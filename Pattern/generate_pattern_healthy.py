import pandas as pd
import os
from datetime import datetime
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
import json 
#mpl.use('TkAgg')

def convert_time(time_str):
    return_time = datetime.strptime(time_str, '[Timestamp(\'%Y-%m-%d %H:%M:%S\')]')
    return return_time

def generate_bins(data_df):
    data_df.loc[data_df['heartrate'].between(0, 50, 'both'), 'hr_bin'] = 'A'
    data_df.loc[data_df['heartrate'].between(50, 60, 'right'), 'hr_bin'] = 'B'
    data_df.loc[data_df['heartrate'].between(60, 70, 'right'), 'hr_bin'] = 'C'
    data_df.loc[data_df['heartrate'].between(70, 80, 'right'), 'hr_bin'] = 'D'
    data_df.loc[data_df['heartrate'].between(80, 90, 'right'), 'hr_bin'] = 'E'
    data_df.loc[data_df['heartrate'].between(90, 100, 'right'), 'hr_bin'] = 'F'
    data_df.loc[data_df['heartrate'].between(100, 110, 'right'), 'hr_bin'] = 'G'
    data_df.loc[data_df['heartrate'].between(110, 120, 'right'), 'hr_bin'] = 'H'
    data_df.loc[data_df['heartrate'].between(120, 130, 'right'), 'hr_bin'] = 'I'
    data_df.loc[data_df['heartrate'].between(130, 140, 'right'), 'hr_bin'] = 'J'
    data_df.loc[data_df['heartrate'].between(140, 150, 'right'), 'hr_bin'] = 'K'
    data_df.loc[data_df['heartrate'].between(150, 200, 'right'), 'hr_bin'] = 'L'

    data_df.loc[data_df['steps'].between(0, 1, 'both'), 'step_bin'] = 'AA'
    data_df.loc[data_df['steps'].between(1, 50, 'right'), 'step_bin'] = 'BB'
    data_df.loc[data_df['steps'].between(50, 500, 'right'), 'step_bin'] = 'CC'
    data_df.loc[data_df['steps'].between(500, 100000, 'right'), 'step_bin'] = 'DD'

    time_of_the_day = []
    index_time = []
    for index, row in data_df.iterrows():
        hour = index.hour
        if hour > 8 and hour < 22:
            time_of_the_day.append('day')
        else:
            time_of_the_day.append('night')
        index_time.append(index)
    data_df['time_of_day'] = pd.Series(time_of_the_day, index=index_time)
    return data_df

noncovid_pattern = []

def generate_pattern(data_df,length_of_segment_hours):
    whole_pattern = data_df['pattern'].tolist()
    pattern_list_segments = []
    for i in range(int(len(whole_pattern)/length_of_segment_hours)):
        if (i+1)*length_of_segment_hours < len(whole_pattern):
            tmp = whole_pattern[i*length_of_segment_hours:(i+1)*length_of_segment_hours]
            pattern_list_segments.append(tmp)
        else:
            pattern_list_segments.append(whole_pattern[i*length_of_segment_hours:])
    return pattern_list_segments

tot_healthy = []
for file in os.listdir('healthy_df/'):
    data_df = pd.read_csv('healthy_df/'+file,index_col=0)
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.resample('1h').mean()
    data = generate_bins(data_df)
    data['pattern'] = data[['hr_bin', 'step_bin','time_of_day']].agg('-'.join, axis=1)
    patient_id = file[:-4]
    # Hyparameter - window size, now is 24 hours
    pattern_list_healthy = generate_pattern(data,24)

    noncovid_pattern = noncovid_pattern + pattern_list_healthy
    tot_healthy.append([[pattern_i for pattern_i in pattern_list_healthy], patient_id])
with open('healthy.json', 'w') as f: 
  json.dump(tot_healthy, f)
from mlxtend.frequent_patterns import apriori

#Hyparameter - minial support
te = TransactionEncoder()
te_ary = te.fit(noncovid_pattern).transform(noncovid_pattern)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets_non_covid = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets_non_covid['length'] = frequent_itemsets_non_covid['itemsets'].apply(lambda x: len(x))

frequent_itemsets_non_covid.to_csv('pattern_healthy30.csv')
