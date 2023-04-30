import numpy as np
import random

covid_data = np.load('covid.npy',allow_pickle=True)
covid_data_aug = np.load('covid_aug.npy',allow_pickle=True)
non_covid = np.load('non_covid.npy',allow_pickle=True)

idx_covid_test = random.sample(list(range(len(covid_data))),int(0.2*len(covid_data)))
idx_noncovid_test = random.sample(list(range(len(non_covid))),int(0.2*len(non_covid)))
test_data_pos = []
for i in range(len(covid_data)):
    if i in idx_covid_test:
        test_data_pos.append(covid_data[i])
test_data_pos = np.concatenate(test_data_pos)
test_label_pos = [1]*len(test_data_pos)

test_data_neg = []
for i in range(len(non_covid)):
    if i in idx_noncovid_test:
        test_data_neg.append(non_covid[i])
test_data_neg = np.concatenate(test_data_neg)
test_label_neg = [0] * len(test_data_neg)
test_data = np.concatenate([test_data_pos,test_data_neg])
test_label = np.array(test_label_pos+test_label_neg)
print(test_data.shape,test_label.shape)
np.save('final_data_cv/x_test.npy',test_data)
np.save('final_data_cv/y_test.npy',test_label)


rest_covid_idx = list(set(list(range(len(covid_data)))) - set(idx_covid_test))
rest_noncovid_idx = list(set(list(range(len(non_covid)))) - set(idx_noncovid_test))

for folder in range(1,6):
    random.seed(folder)
    idx_covid = random.sample(rest_covid_idx,int(0.2*len(covid_data)))
    idx_noncovid = random.sample(rest_noncovid_idx,int(0.2*len(non_covid)))

    val_data_pos = []
    train_data_pos  = []
    for i in range(len(covid_data)):
        if i in idx_covid:
            val_data_pos.append(covid_data[i])
        else:
            train_data_pos.append(covid_data_aug[i])
            # train_data_pos.append(covid_data[i])
    val_data_pos = np.concatenate(val_data_pos)
    train_data_pos = np.concatenate(train_data_pos)
    train_label_pos = [1]*len(train_data_pos)
    val_label_pos = [1]*len(val_data_pos)
    val_data_neg = []
    train_data_neg  = []
    for i in range(len(non_covid)):
        if i in idx_noncovid:
            val_data_neg.append(non_covid[i])
        else:
            train_data_neg.append(non_covid[i])
    val_data_neg = np.concatenate(val_data_neg)
    train_data_neg = np.concatenate(train_data_neg)
    train_label_neg = [0]*len(train_data_neg)
    val_label_neg = [0]*len(val_data_neg)

    train_data = np.concatenate([train_data_pos,train_data_neg])
    val_data = np.concatenate([val_data_pos,val_data_neg])
    train_label = np.array(train_label_pos+train_label_neg)
    val_label = np.array(val_label_pos+val_label_neg)
    print(train_data.shape, train_label.shape,val_data.shape,val_label.shape)
    np.save('final_data_cv/x_train%d.npy'%folder,train_data)
    np.save('final_data_cv/y_train%d.npy'%folder,train_label)
    np.save('final_data_cv/x_val%d.npy'%folder,val_data)
    np.save('final_data_cv/y_tval%d.npy'%folder,val_label)