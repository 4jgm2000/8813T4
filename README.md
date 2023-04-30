# 8813T4
Team 4 project for 8813 Covid detection using wearable data

Overleaf - https://www.overleaf.com/5833861397jzvfyfgkrdtv



#LSTM Model


#Preprocessing Files (generate_pattern_healthy.py generate_pattern.py, generate_pattern_sleep.py)
These files process the data from Mishra et al. by loading, interpolating, and binning smartwatch data into pattern labels for each group (healthy, Covid, and those with sleep data)
Each pattern sequence specifically for the Covid group is then divided into healthy or Covid sequences.
Uses helper function load_data.py and load_healthy_data.py

#Generate Patterns (test_train_split.ipynb)
Will generate patterns for specific groups (pattern_covidxx.csv, pattern_healthyxx.csv; where xx refers to the model support patterns). Requires .json from generate_pattern_healthy.py generate_pattern.py and generate_pattern_sleep.py
This file also creates the train, test, and external validation sets we use in the future model steps. 

#Pattern Classifier - Pattern/patter_classifier.py
Will read in csv based support patterns and find difference. Will then do NB on the 5 cv, no further parameters. 

