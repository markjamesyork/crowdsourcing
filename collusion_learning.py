# collusion_learning.py
# please use Python >=3.9

from enum import Enum, auto
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, ReLU, LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import random
import scipy.stats as scipystats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler
from statistics import NormalDist
import tensorflow as tf
from typing import Optional, Union

THRESHOLD = 0.1
N = 5
M = 6
EPSILON = 0.001

N_TEST_CASES = 500
PROBS = tf.random.uniform([M], 0.1, 0.9)
print(PROBS)


def calculate_profit(reports):
    # have to code simulation in tensorflow in differentiable way
    reports = tf.reshape(reports, [N, M])
    reports = tf.clip_by_value(reports, EPSILON, 1 - EPSILON)
    # reports += EPSILON
    weights = tf.fill([N], 1/N)
    beliefs = tf.linalg.matvec(reports, weights, transpose_a=True)
    allocation = tf.greater(beliefs, THRESHOLD)
    expected_outcomes = tf.where(allocation, PROBS, 0)

    # tf.print(expected_outcomes)

    min_reports = (THRESHOLD - (tf.tile(tf.reshape(beliefs, [1, M]), [N, 1]) - (reports *
                   tf.tile(tf.reshape(weights, [N, 1]), [1, M])))) * tf.tile(1 / tf.reshape(weights, [N, 1]), [1, M])
    min_reports = tf.clip_by_value(min_reports, EPSILON, 1 - EPSILON)

    payment_indicators = tf.cast(tf.greater(reports, min_reports), tf.float32)

    payments_repaid = expected_outcomes * \
        (tf.math.log(reports) - tf.math.log(min_reports)) / \
        (-1 * tf.math.log(min_reports))
    payments_not_repaid = (1 - expected_outcomes) * (tf.math.log(1 - reports) -
                                                     tf.math.log(1 - min_reports)) / (-1 * tf.math.log(min_reports))
    outcome_payments = payment_indicators * \
        (payments_repaid + payments_not_repaid)
    return -1 * tf.math.reduce_sum(outcome_payments)


def custom_loss(X):

    def loss(y_true, y_pred):
        # ignore y_true since we don't know it
        # y_pred is the learned reports

        reports = tf.reshape(y_pred, [tf.size(y_pred) / (N * M), N * M])
        result = tf.map_fn(lambda report: calculate_profit(report), reports)
        return tf.math.reduce_sum(result)

    return loss


def main() -> None:
    X = np.random.random((N_TEST_CASES, N * M))
    # X = np.full((N_TEST_CASES, N * M), 0.5)
    y = np.zeros((N_TEST_CASES, N * M))

    inputs = Input(shape=(N * M, ))
    layer = Dense(N * M)(inputs)
    layer = Dense(N * M)(layer)
    outputs = Dense(N * M)(layer)

    # layer = LeakyReLU(alpha=0.05)(layer)
    # outputs = ReLU()(layer)

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    model.compile(loss=custom_loss(X),
                  optimizer='adam')
    history = model.fit(X, y, validation_split=0.2,
                        epochs=100, batch_size=16, verbose=0)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    predictions = model.predict(np.full((1, N * M), 0.5))
    print(predictions.reshape((N, M)))


if __name__ == '__main__':
    main()
