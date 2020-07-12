import utils_preprocessing as upp
import utils_vectorisation as uv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
import pandas as pd
import numpy as np
import os

data_directory = os.environ['DATA']



X_train = np.load(f'{data_directory}/pol_seq_bow_X.npy')
Y_train = np.load(f'{data_directory}/pol_seq_bow_Y.npy')


print('The dimensions of X_train are: ', X_train.shape)
print('The dimensions of Y_train are: ', Y_train.shape)

# set up iterations for the hidden layers and learning rates

hidden_layer_1 = [16, 32]
hidden_layer_2 = [32]
hidden_layer_3 = [0]
learning_rates = [0.0003]

for lr in learning_rates:
    for hl1 in hidden_layer_1:
        for hl2 in hidden_layer_2:
            for hl3 in hidden_layer_3:
                for i in range(2):
                    # declare name of network for storage
                    name = f'seq_bow_adam_rc_0.2_lr_{lr}_hl1_{hl1}_hl2_{hl2}_hl3_{hl3}_run_{i}'
                    # initialise Tensorboard object and directory
                    tensorboard = TensorBoard(log_dir=f'pol_training/logs_seq_bow/{name}')
                    # declare path for training history to be delivered to
                    csv_file = f'pol_training/results_seq_bow/training_logs/{name}_training_log.csv'
                    csv_logger = CSVLogger(csv_file, separator=',', append=False)

                    # instantiate model
                    model = tf.keras.Sequential()
                    if hl2 == 0:
                        model.add(LSTM(hl1, input_shape=(X_train.shape[1:]), activation='tanh', use_bias=True, recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2))
                    else:
                        model.add(LSTM(hl1, input_shape=(X_train.shape[1:]), activation='tanh', use_bias=True, recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
                        model.add(LSTM(hl2, input_shape=(X_train.shape[1:]), activation='tanh', use_bias=True, recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2))
                    if hl3 != 0:
                        model.add(Dense(hl3, activation='relu'))
                        model.add(Dropout(0.2))
                    model.add(Dense(1, activation='sigmoid'))
                    # compile
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                    # fit and cache history
                    history = model.fit(X_train, Y_train, batch_size=64, epochs=30, validation_split=0.2, callbacks=[tensorboard, csv_logger])
