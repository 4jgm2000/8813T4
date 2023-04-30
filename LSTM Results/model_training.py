import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Flatten, Dense,LSTM
from tensorflow.keras.models import Model
import numpy as np
from keras.utils import np_utils
import os
x_test = np.load('final_data_cv/x_test.npy')
y_test = np.load('final_data_cv/y_test.npy')
TIME_STEPS = 8
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
for fold in range(1,6):
    x_train = np.load('final_data_cv/x_train%d.npy'%fold)
    x_val = np.load('final_data_cv/x_val%d.npy'%fold)
    y_train = np.load('final_data_cv/y_train%d.npy'%fold)
    y_val = np.load('final_data_cv/y_val%d.npy'%fold)

    weight_path = 'model_weights/'
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    # early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode="min")
    checkpoint = keras.callbacks.ModelCheckpoint((weight_path+'/model.{epoch:02d}.hdf5'), verbose=1, monitor='val_acc',save_best_only=True, mode='auto')

    # Set the input shape
    input_shape = (8, 1)
     # Define the model
    model = Sequential()

    # Add the LSTM layers
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=16))
    model.add(keras.layers.Dropout(rate=0.2))
    # Add the output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Define model
    # model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train,y_train, validation_data=(x_val, y_val),
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      shuffle=True,
      callbacks=[checkpoint])

from sklearn.preprocessing import binarize
import os
from sklearn.metrics import roc_curve, auc

x_val = np.load('final_data_cv/x_val1.npy')
y_val = np.load('final_data_cv/y_val1.npy')
for weight_file in os.listdir(weight_path):
    if '.hdf5' in weight_file:
      model.load_weights(weight_path+'/'+weight_file)
      # print(weight_file)
      # get optimal threshold from validation set
      y_pred_val = model.predict(x_val, verbose=0)
      fpr, tpr, thresholds = roc_curve(y_val, y_pred_val)
      optimal_idx = np.argmax(tpr - fpr)
      optimal_threshold = thresholds[optimal_idx]
      
      #predict on test set
      y_pred = model.predict(x_test,verbose=0)
      y_pred = binarize(y_pred,threshold = optimal_threshold)
      from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,f1_score
      c = confusion_matrix(y_test,y_pred)
      # print('Confusion matrix:\n', c)
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
      sensitivity  = tp / (tp+fn)
      specificity = tn / (tn+fp)

      print(sensitivity,'\t',specificity)
