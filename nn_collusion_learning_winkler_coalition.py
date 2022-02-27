# nn_collusion_learning_winkler_coalition.py
# please use Python >=3.9

from enum import Enum, auto
import itertools
import matplotlib.pyplot as plt
import numpy as np
# import numpy.typing as npt
import random
import scipy.stats as scipystats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler
from statistics import NormalDist
import sys
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.framework.ops import disable_eager_execution
from typing import Optional, Union

# other files
# from sgdr import SGDR

# disable_eager_execution()

THRESHOLD = 0.1
N_COALITION = 3
N_TOTAL = N_COALITION + 7
M = 6
EPSILON = 0.001

PROBS = tf.convert_to_tensor([0.24, 0.41, 0.53, 0.67, 0.72, 0.79])

BATCH_SIZE = 32
N_TEST_CASES = BATCH_SIZE * 32


# UTIL FUNCTIONS

def sigmoid(x, threshold, steepness=70):
    return tf.math.exp(steepness*(x-threshold)) / (1 + tf.math.exp(steepness*(x-threshold)))


def all_binary_strings(n):
    '''returns all binary strings of length n'''
    return np.array([i for i in itertools.product([0, 1], repeat=n)])


# LOSS FUNCTIONS

def calculate_loss_in_profit(reports):
    # have to code simulation in tensorflow in differentiable way
    coalition_reports = tf.reshape(reports, [N_COALITION, M])
    other_reports = tf.tile(tf.reshape(PROBS, [1, M]), [
                            N_TOTAL - N_COALITION, 1])
    reports = tf.concat([coalition_reports, other_reports], axis=0)
    reports = tf.clip_by_value(reports, EPSILON, 1 - EPSILON)
    weights = tf.fill([N_TOTAL], 1/N_TOTAL)
    beliefs = tf.linalg.matvec(reports, weights, transpose_a=True)
    allocation = tf.greater(beliefs, THRESHOLD)
    expected_outcomes = tf.where(allocation, PROBS, 0)

    min_reports = (THRESHOLD - (tf.tile(tf.reshape(beliefs, [1, M]), [N_TOTAL, 1]) - (reports *
                   tf.tile(tf.reshape(weights, [N_TOTAL, 1]), [1, M])))) * tf.tile(1 / tf.reshape(weights, [N_TOTAL, 1]), [1, M])
    min_reports = tf.clip_by_value(min_reports, EPSILON, 1 - EPSILON)

    payments_repaid = expected_outcomes * \
        (tf.math.log(reports) - tf.math.log(min_reports)) / \
        (-1 * tf.math.log(min_reports))
    payments_not_repaid = (1 - expected_outcomes) * (tf.math.log(1 - reports) -
                                                     tf.math.log(1 - min_reports)) / (-1 * tf.math.log(min_reports))
    outcome_payments = tf.where(tf.greater(
        reports, min_reports), payments_repaid + payments_not_repaid, 0)

    coalition_outcome_payments = outcome_payments[:N_COALITION, :]
    return -1 * tf.math.reduce_sum(coalition_outcome_payments)


def profit_loss(y_true, y_pred):
    # ignore y_true since we don't know it
    # y_pred is the learned reports
    reports = tf.reshape(
        y_pred, [tf.size(y_pred) / (N_COALITION * M), N_COALITION * M])
    result = tf.map_fn(
        lambda report: calculate_loss_in_profit(report), reports)
    return tf.math.reduce_sum(result)


def calculate_loss_in_profit_sigmoid(reports):
    # have to code simulation in tensorflow in differentiable way
    coalition_reports = tf.reshape(reports, [N_COALITION, M])
    other_reports = tf.tile(tf.reshape(PROBS, [1, M]), [
                            N_TOTAL - N_COALITION, 1])
    reports = tf.concat([coalition_reports, other_reports], axis=0)
    reports = tf.clip_by_value(reports, EPSILON, 1 - EPSILON)
    # reports += EPSILON
    weights = tf.fill([N_TOTAL], 1/N_TOTAL)
    beliefs = tf.linalg.matvec(reports, weights, transpose_a=True)
    # use a sigmoid function instead of piecewise 0 for differentiability
    expected_outcomes = PROBS * sigmoid(beliefs, THRESHOLD)

    min_reports = (THRESHOLD - (tf.tile(tf.reshape(beliefs, [1, M]), [N_TOTAL, 1]) - (reports *
                   tf.tile(tf.reshape(weights, [N_TOTAL, 1]), [1, M])))) * tf.tile(1 / tf.reshape(weights, [N_TOTAL, 1]), [1, M])
    min_reports = tf.clip_by_value(min_reports, EPSILON, 1 - EPSILON)

    payments_repaid = expected_outcomes * \
        (tf.math.log(reports) - tf.math.log(min_reports)) / \
        (-1 * tf.math.log(min_reports))
    payments_not_repaid = (1 - expected_outcomes) * (tf.math.log(1 - reports) -
                                                     tf.math.log(1 - min_reports)) / (-1 * tf.math.log(min_reports))
    # again sigmoid for differentiability
    outcome_payments = (payments_repaid + payments_not_repaid) * \
        sigmoid(reports, min_reports)

    coalition_outcome_payments = outcome_payments[:N_COALITION, :]
    return -1 * tf.math.reduce_sum(coalition_outcome_payments)


def profit_loss_sigmoid(y_true, y_pred):
    # ignore y_true since we don't know it
    # y_pred is the learned reports
    reports = tf.reshape(
        y_pred, [tf.size(y_pred) / (N_COALITION * M), N_COALITION * M])
    result = tf.map_fn(
        lambda report: calculate_loss_in_profit_sigmoid(report), reports)
    return tf.math.reduce_sum(result)


def calculate_loss_in_desired_borrowers(reports, desiderata):
    # have to code simulation in tensorflow in differentiable way
    coalition_reports = tf.reshape(reports, [N_COALITION, M])
    other_reports = tf.tile(tf.reshape(PROBS, [1, M]), [
                            N_TOTAL - N_COALITION, 1])
    reports = tf.concat([coalition_reports, other_reports], axis=0)
    reports = tf.clip_by_value(reports, EPSILON, 1 - EPSILON)
    desiderata = tf.reshape(desiderata, [N_COALITION, M])
    weights = tf.fill([N_TOTAL], 1/N_TOTAL)
    beliefs = tf.linalg.matvec(reports, weights, transpose_a=True)
    # sigmoid instead of step function for differentiability
    allocation = sigmoid(beliefs, THRESHOLD, 5)
    desirability_utilities = desiderata * \
        tf.tile(tf.reshape(allocation, [1, M]), [N_COALITION, 1])
    return -1 * tf.math.reduce_sum(desirability_utilities)


def desirability_loss(y_true, y_pred):
    # y_true is ignored
    # y_pred is the learned reports concatenated with the inputs (how much each recommender cares about each borrower)
    reports, desiderata = tf.split(y_pred, 2, axis=1)
    reports = tf.reshape(
        reports, [tf.size(reports) / (N_COALITION * M), N_COALITION * M])
    desiderata = tf.reshape(
        desiderata, [tf.size(desiderata) / (N_COALITION * M), N_COALITION * M])
    result = tf.map_fn(
        lambda x: calculate_loss_in_desired_borrowers(x[0], x[1]), (reports, desiderata), dtype=tf.float32)
    return tf.math.reduce_sum(result)


def mixed_loss(desirability_importance=0.5):

    def desirability_and_profit_loss(y_true, y_pred):
        reports, _ = tf.split(y_pred, 2, axis=1)
        return desirability_importance * desirability_loss(y_true, y_pred) + \
            (1-desirability_importance) * profit_loss_sigmoid(y_true,
                                                              reports)  # need to remove desiderata for profit calc

    return desirability_and_profit_loss


# NEURAL NETWORKS

def profit_reports() -> None:
    # here we are just maximizing profit from the mechanism, resulting in accurate reports
    # X doesn't do anything, it's just stochastic noise for the network to run
    X = np.random.random((N_TEST_CASES, N_COALITION * M))
    # X = np.full((N_TEST_CASES, N * M), 0.5)
    # y is also ignored
    y = np.zeros((N_TEST_CASES, 1))

    inputs = Input(shape=(N_COALITION * M, ))
    layer = Dense(N_COALITION * M)(inputs)
    layer = Dense(N_COALITION * M)(layer)
    outputs = Dense(N_COALITION * M, activation='sigmoid')(layer)

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    # sgdr = SGDR(min_lr=1e-5, max_lr=0.01)
    model.compile(loss=profit_loss,
                  optimizer=Adam(amsgrad=True))
    # optimizer=SGD(learning_rate=1e-4, momentum=0.1))
    history = model.fit(X, y, validation_split=0.2,
                        epochs=100, batch_size=BATCH_SIZE, verbose=0)
    # callbacks=[sgdr])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    predictions = model.predict(np.full((1, N_COALITION * M), 0.5))
    predictions = predictions.reshape((N_COALITION, M))
    print('mean: ' + str(np.mean(predictions, axis=0)))
    print('std: ' + str(np.std(predictions, axis=0)))
    print('')


def profit_reports_sigmoid() -> None:
    # here we are just maximizing profit from the mechanism, resulting in accurate reports
    # X doesn't do anything, it's just stochastic noise for the network to run
    X = np.random.random((N_TEST_CASES, N_COALITION * M))
    # X = np.full((N_TEST_CASES, N * M), 0.5)
    # y is also ignored
    y = np.zeros((N_TEST_CASES, 1))

    inputs = Input(shape=(N_COALITION * M, ))
    layer = Dense(N_COALITION * M)(inputs)
    layer = Dense(N_COALITION * M)(layer)
    outputs = Dense(N_COALITION * M, activation='sigmoid')(layer)

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    # sgdr = SGDR(min_lr=1e-5, max_lr=0.05)
    model.compile(loss=profit_loss_sigmoid,
                  optimizer=Adam(amsgrad=True))
    #   optimizer=SGD(learning_rate=1e-4, momentum=0.9))
    history = model.fit(X, y, validation_split=0.2,
                        epochs=100, batch_size=BATCH_SIZE, verbose=0)
    # callbacks=[sgdr])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    predictions = model.predict(np.full((1, N_COALITION * M), 0.5))
    predictions = predictions.reshape((N_COALITION, M))
    print('mean: ' + str(np.mean(predictions, axis=0)))
    print('std: ' + str(np.std(predictions, axis=0)))
    print('')


def desire_borrowers() -> None:
    # in this case the quantity that we are maximizing is actually the probabilities
    # that the desired people get loans
    # here, x_{i,q} represents how much that recommender i wants q to get a loan.

    # when the coalition all agrees on one borrower
    # indices = np.random.randint(0, M, N_TEST_CASES)
    # X = np.array([np.tile([int(j == index) for j in range(M)], N)
    #              for _, index in enumerate(indices)], dtype=np.float32)

    X = np.random.randint(0, 2, (N_TEST_CASES, N_COALITION * M))

    # y is ignored
    y = np.zeros((N_TEST_CASES, N_COALITION * M))

    inputs = Input(shape=(N_COALITION * M, ), dtype=tf.float32)
    layer = Dense(N_COALITION * M)(inputs)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M, activation='sigmoid')(layer)
    outputs = Concatenate()([layer, inputs])

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    model.compile(loss=desirability_loss,
                  optimizer=Adam(amsgrad=True))
    history = model.fit(X, y, validation_split=0.2,
                        epochs=50, batch_size=BATCH_SIZE, verbose=0)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    # iterate over borrowers that recommenders care about
    tests = []
    for index in range(M):
        test = np.zeros(M)
        test[index] = 1
        test = np.tile(test, N_COALITION)
        tests.append(test)
    predictions = model.predict(np.array(tests))
    predictions, _ = np.split(predictions, 2, axis=1)
    for prediction in predictions:
        prediction = prediction.reshape((N_COALITION, M))
        print('mean: ' + str(np.mean(prediction, axis=0)))
        print('std: ' + str(np.std(prediction, axis=0)))
        print('')


def desire_borrowers_and_profit(alpha=0.5) -> None:
    # in this case recommenders care about both profits and helping their desired borrower get a loan

    # here, x_{i,q} represents how much that recommender i wants q to get a loan.
    # indices = np.random.randint(0, M, N_TEST_CASES)
    # X = np.array([np.tile([int(j == index) for j in range(M)], N)
    #              for _, index in enumerate(indices)], dtype=np.float32)

    X = np.random.randint(0, 2, (N_TEST_CASES, N_COALITION * M))

    # y is ignored
    y = np.zeros((N_TEST_CASES, N_COALITION * M))

    inputs = Input(shape=(N_COALITION * M, ), dtype=tf.float32)
    layer = Dense(N_COALITION * M)(inputs)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M, activation='sigmoid')(layer)
    outputs = Concatenate()([layer, inputs])

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    # the mixed loss coefficient is what percent the recommenders care about their desired borrower
    # as compared to profit
    model.compile(loss=mixed_loss(alpha),
                  optimizer=Adam(amsgrad=True))
    history = model.fit(X, y, validation_split=0.2,
                        epochs=300, batch_size=BATCH_SIZE, verbose=0)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    # iterate over borrowers that recommenders care about
    tests = []
    for index in range(M):
        test = np.zeros(M)
        test[index] = 1
        test = np.tile(test, N_COALITION)
        tests.append(test)
    predictions = model.predict(np.array(tests))
    predictions, _ = np.split(predictions, 2, axis=1)
    for prediction in predictions:
        prediction = prediction.reshape((N_COALITION, M))
        print('mean: ' + str(np.mean(prediction, axis=0)))
        print('std: ' + str(np.std(prediction, axis=0)))
        print('')


def main() -> None:
    profit_reports()
    profit_reports_sigmoid()
    desire_borrowers()
    desire_borrowers_and_profit(0.5)
    desire_borrowers_and_profit(0.3)


if __name__ == '__main__':
    main()
