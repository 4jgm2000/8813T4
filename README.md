# 8813T4
Team 4 project for 8813 Covid detection using wearable data

#LSTM Model


#Preprocessing Files (generate_pattern_healthy.py generate_pattern.py, generate_pattern_sleep.py)
These files process the data from Mishra et al. by iterpolating and binning data into pattern labels. 

#Generate Patterns (test_train_split.ipynb)
Will generate patterns for specific groups. Requires .json from generate_pattern_healthy.py generate_pattern.py and generate_pattern_sleep.py
This file also creates the train, test, and external validation sets we use in the future model steps. 

#Pattern Classifier
Will read in csv based support patterns and find difference. Will then do NB on the 5 cv, no further parameters. 

