import utils_preprocessing as upp
import utils_vectorisation as uv
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
import os
import numpy as np
import pandas as pd

data_directory = os.environ['DATA']


X_train = np.load(f'{data_directory}/pol_emb_X.npy')
Y_train = np.load(f'{data_directory}/pol_emb_Y.npy')


print('The dimensions of X_train are: ', X_train.shape)
print('The dimensions of Y_train are: ', Y_train.shape)


# set up iterations for the hidden layers and learning rates

hidden_layer_1 = [128]
hidden_layer_2 = [32]
hidden_layer_3 = [32]
learning_rates = [0.0003]


# loop through learning rates and and hidden layers unit quantities training three
# networks per orientation

for lr in learning_rates:
    for hl1 in hidden_layer_1:
        for hl2 in hidden_layer_2:
            for hl3 in hidden_layer_3:
                for i in range(2):
                    # declare name of network for storage
                    name = f'emb200_adam_rc_0.4_lr_{lr}_hl1_{hl1}_hl2_{hl2}_hl3_{hl3}_run_{i}'
                    # initialise Tensorboard object and directory
                    tensorboard = TensorBoard(log_dir=f'pol_training/logs_emb/{name}')
                    # declare path for training history to be delivered to
                    csv_file = f'pol_training/results_emb/training_logs/{name}_training_log.csv'
                    csv_logger = CSVLogger(csv_file, separator=',', append=False)

                    # instantiate model
                    model = tf.keras.Sequential()
                    model.add(layers.Dense(hl1, activation='relu'))
                    model.add(layers.Dropout(0.4))
                    if hl2 != 0:
                        model.add(layers.Dense(hl2, activation='relu'))
                        model.add(layers.Dropout(0.4))
                        if (hl2 != 0) and (hl3 != 0):
                            model.add(layers.Dense(hl3, activation='relu'))
                            model.add(layers.Dropout(0.4))
                    model.add(layers.Dense(1, activation='sigmoid'))
                    # compile
                    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                    # fit and cache history
                    history = model.fit(X_train, Y_train, batch_size=64, epochs=700, validation_split=0.2, callbacks=[tensorboard, csv_logger])
