""" ANN """
#!/usr/bin/env python

# pylint: disable=W0511 # suppress TODOs

from datetime import datetime
from joblib import dump, load
import numpy as np
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras import activations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ANN:
    """ ANN class implementing neural network IK approach """
    def __init__(self, effector_workspace_limits, dh_matrix):
        self.effector_workspace_limits = effector_workspace_limits
        self.dh_matrix = dh_matrix
        self.model = None
        self.x_data_skaler = StandardScaler()
        self.y_data_skaler = StandardScaler()

    def __fit_trainig_data(self, samples, features):
        """ Split training/test (70/30) data and use StandardScaler to scale it """
        x_train, x_test, y_train, y_test = \
            train_test_split(samples, features, test_size=0.33, random_state=42)

        x_train = self.x_data_skaler.fit_transform(x_train)
        x_test = self.x_data_skaler.transform(x_test)

        y_train = self.y_data_skaler.fit_transform(y_train)
        y_test = self.y_data_skaler.transform(y_test)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    def train_model(self, epochs, samples, features):
        """ Train ANN Sequential model """
        self.model = Sequential()

        data_in, data_out, data_test_in, data_test_out = self.__fit_trainig_data(samples, features)

        self.model.add(Input(shape=(3,))) # Input layer, 3 input variables

        net_shape = [
                (12, 500, activations.tanh)
            ]

        for shape in net_shape:
            for _ in range(shape[0]):
                self.model.add(Dense(units=shape[1], activation=shape[2])) # hidden layer

        self.model.add(Dense(units=4)) # theta1, theta2, theta3, theta4 -> output layer

        early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

        self.model.compile(optimizer = Adam(learning_rate=1.0e-5), loss='mse')

        self.model.fit(
                        data_in, data_out,
                        validation_data = (data_test_in, data_test_out),
                        epochs = epochs,
                        callbacks = [early_stopping],
                        # batch_size=64
                      )

    def predict(self, position):
        """ Use trained ANN to predict joint angles, scale input and rescale output """
        predictions = self.y_data_skaler.inverse_transform(
            self.model.predict(self.x_data_skaler.transform(position))
        )

        return predictions

    def load_model(self, model_h5):
        """ Load model from file """
        self.model = load_model(model_h5)
        modelname = model_h5[:-3]
        # load scalers for this model
        self.x_data_skaler = load(f'{modelname}_scaler_x.bin')
        self.y_data_skaler = load(f'{modelname}_scaler_y.bin')
        return self.model

    def save_model(self, prefix = 'model'):
        """ Save model to file """
        date_now = datetime.now()
        # replace . with - in filename to look better
        timestamp_str = str(datetime.timestamp(date_now)).replace('.','-')
        self.model.save(f'{prefix}_{timestamp_str}.h5')
        # save scalers
        dump(self.x_data_skaler, f'{prefix}_{timestamp_str}_scaler_x.bin', compress=True)
        dump(self.y_data_skaler, f'{prefix}_{timestamp_str}_scaler_y.bin', compress=True)
