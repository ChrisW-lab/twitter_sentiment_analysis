import os
print(os.environ)

import utils_preprocessing as upp
import utils_vectorisation as uv
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import TensorBoard, CSVLogger

import numpy as np
import pandas as pd


data_directory = os.environ['DATA']


X_train = np.load(f'{data_directory}/pol_bow_X.npy')
Y_train = np.load(f'{data_directory}/pol_bow_Y.npy')

print('X_train shape is: ', X_train.shape)
print('Y_train.shape is: ', Y_train.shape)

# set up iterations for the hidden layers and learning rates

hidden_layer_1 = [7, 9, 10, 11, 12, 13, 16, 32, 64]
hidden_layer_2 = [0]
hidden_layer_3 = [0]
learning_rates = [0.0003]


# loop through learning rates and and hidden layers unit quantities training three
# networks per orientation

for lr in learning_rates:
    for hl1 in hidden_layer_1:
        for hl2 in hidden_layer_2:
            for hl3 in hidden_layer_3:
                for i in range(2):
                    # declare name of network for storage
                    name = f'bow_adam_lr_{lr}_hl1_{hl1}_hl2_{hl2}_hl3_{hl3}_run_{i}'
                    # initialise Tensorboard object and directory
                    tensorboard = TensorBoard(log_dir=f'pol_training/logs_bow/{name}')
                    # declare path for training history to be delivered to
                    csv_file = f'pol_training/results_bow/training_logs/{name}_training_log.csv'
                    csv_logger = CSVLogger(csv_file, separator=',', append=False)

                    # instantiate model
                    model = tf.keras.Sequential()
                    model.add(layers.Dense(hl1, activation='relu'))
                    model.add(layers.Dropout(0.2))
                    if hl2 != 0:
                        model.add(layers.Dense(hl2, activation='relu'))
                        model.add(layers.Dropout(0.2))
                        if hl3 != 0:
                            model.add(layers.Dense(hl3, activation='relu'))
                            model.add(layers.Dropout(0.2))
                    model.add(layers.Dense(1, activation='sigmoid'))
                    # compile
                    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                    # fit and cache history
                    history = model.fit(X_train, Y_train, batch_size=64, epochs=120, validation_split=0.2, callbacks=[tensorboard, csv_logger])
