import pandas as pd
import os
from datetime import datetime
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
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

f = open('covid.txt').read()
count = 0
patient_dates = {}
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
    patient_dates[filename] = [symtom_date,recovery_date]

covid_pattern = []
noncovid_pattern = []
covid_names = []
noncovid_names = []

tot_covid = []
tot_noncovid_post = []
tot_noncovid_pre = []

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

for file in os.listdir('covid_df/'):
    data_df = pd.read_csv('covid_df/'+file,index_col=0)
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.resample('1h').mean()
    data = generate_bins(data_df)
    data['pattern'] = data[['hr_bin', 'step_bin','time_of_day']].agg('-'.join, axis=1)

    patient_id = file[:-4]
    time1 = patient_dates[patient_id][0]
    time2 = patient_dates[patient_id][1]
    # Hyparameter - control the time range between symtom and recovery
    time1 = time1 + dt.timedelta(days=0)
    time2 = time2 - dt.timedelta(days=0)
    data_before = data[:time1]
    data_illness = data[time1:time2]
    date_recovery = data[time2:]
    # Hyparameter - window size, now is 24 hours
    pattern_list_before = generate_pattern(data_before,24)
    pattern_list_illness= generate_pattern(data_illness, 24)
    pattern_list_recovery = generate_pattern(date_recovery,24)

    covid_pattern = covid_pattern + pattern_list_illness
    noncovid_pattern = noncovid_pattern + pattern_list_before + pattern_list_recovery
    
    tot_covid.append([[pattern_i for pattern_i in pattern_list_illness], patient_id])
    tot_noncovid_pre.append([[pattern_i for pattern_i in pattern_list_before], patient_id])
    tot_noncovid_post.append([[pattern_i for pattern_i in pattern_list_recovery], patient_id])
with open('covid.json', 'w') as f: 
  json.dump(tot_covid, f)
with open('noncovid_pre.json', 'w') as f: 
  json.dump(tot_noncovid_pre, f)
with open('noncovid_post.json', 'w') as f: 
  json.dump(tot_noncovid_post, f)
te = TransactionEncoder()
te_ary = te.fit(covid_pattern).transform(covid_pattern)
df = pd.DataFrame(te_ary, columns=te.columns_)
from mlxtend.frequent_patterns import apriori

#Hyparameter - minial support
frequent_itemsets_covid = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets_covid['length'] = frequent_itemsets_covid['itemsets'].apply(lambda x: len(x))

# 
te = TransactionEncoder()
te_ary = te.fit(noncovid_pattern).transform(noncovid_pattern)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets_non_covid = apriori(df, min_support=0.4, use_colnames=True)
frequent_itemsets_non_covid['length'] = frequent_itemsets_non_covid['itemsets'].apply(lambda x: len(x))


frequent_covid_list = frequent_itemsets_covid['itemsets'].tolist()
print(frequent_covid_list)
frequent_itemsets_covid.to_csv('pattern_covid50.csv')
frequent_noncovid_list = frequent_itemsets_non_covid['itemsets'].tolist()

frequent_healthy= pd.read_csv('pattern_healthy30.csv')
print(frequent_healthy['itemsets'])
frequent_healthy_list = frequent_healthy['itemsets'].tolist()
print(list(set(frequent_covid_list).difference(set(frequent_healthy_list))))
# frequent_itemsets_covid.to_csv('pattern_covid.csv')
# frequent_itemsets_non_covid.to_csv('pattern_noncovid.csv')
