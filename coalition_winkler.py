# nn_collusion_learning_winkler_coalition.py
# please use Python >=3.9

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import stack
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
from keras.utils.vis_utils import plot_model

PRINT_COMMENTS = True


THRESHOLD = 0.5
N_COALITION = 3
N_TOTAL = 6
M = 4
EPSILON = 0.001

PROBS = [0.43, 0.62, 0.70, 0.76]
assert len(PROBS) == M
PROBS = tf.convert_to_tensor(PROBS)

BATCH_SIZE = 32
N_TEST_CASES = BATCH_SIZE * 128


# UTIL FUNCTIONS

def sigmoid(x, threshold, steepness=70):
    return tf.math.exp(steepness*(x-threshold)) / (1 + tf.math.exp(steepness*(x-threshold)))


def all_binary_strings(n):
    '''returns all binary strings of length n'''
    return np.array([i for i in itertools.product([0, 1], repeat=n)])


# LOSS FUNCTIONS

def calculate_loss_in_profit(outcomes, reports):
    # have to code simulation in tensorflow in differentiable way
    coalition_reports = tf.reshape(reports, [N_COALITION, M])
    other_reports = tf.tile(tf.reshape(PROBS, [1, M]), [
                            N_TOTAL - N_COALITION, 1])
    reports = tf.concat([coalition_reports, other_reports], axis=0)
    reports = tf.clip_by_value(reports, EPSILON, 1 - EPSILON)
    weights = tf.fill([N_TOTAL], 1/N_TOTAL)
    beliefs = tf.linalg.matvec(reports, weights, transpose_a=True)
    allocation = tf.greater(beliefs, THRESHOLD)
    outcomes = tf.cast(tf.where(allocation, outcomes, 0), dtype=tf.float32)
    min_reports = (THRESHOLD - (tf.tile(tf.reshape(beliefs, [1, M]), [N_TOTAL, 1]) - (reports *
                   tf.tile(tf.reshape(weights, [N_TOTAL, 1]), [1, M])))) * tf.tile(1 / tf.reshape(weights, [N_TOTAL, 1]), [1, M])
    min_reports = tf.clip_by_value(min_reports, EPSILON, 1 - EPSILON)

    payments_repaid = outcomes * \
        (tf.math.log(reports) - tf.math.log(min_reports)) / \
        (-1 * tf.math.log(min_reports))
    payments_not_repaid = (1 - outcomes) * (tf.math.log(1 - reports) -
                                            tf.math.log(1 - min_reports)) / (-1 * tf.math.log(min_reports))
    # sigmoid for differentiability
    outcome_payments = tf.where(tf.greater(
        reports, min_reports), payments_repaid + payments_not_repaid, 0)
    coalition_outcome_payments = outcome_payments[:N_COALITION, :]
    return -1 * tf.math.reduce_sum(coalition_outcome_payments)


def profit_loss(y_true, y_pred):
    # y_true is the binary outcome if allocated to the borrower; training data should have
    # the binary outcome ~ true probability of repayment for all borrowers
    # y_pred is the learned reports
    result = tf.map_fn(
        lambda x: calculate_loss_in_profit(x[0], x[1]), (y_true, y_pred), fn_output_signature=tf.float32)
    return tf.math.reduce_sum(result)


def calculate_loss_in_desired_borrowers(reports, preferences):
    # have to code simulation in tensorflow in differentiable way
    coalition_reports = tf.reshape(reports, [N_COALITION, M])
    other_reports = tf.tile(tf.reshape(PROBS, [1, M]), [
                            N_TOTAL - N_COALITION, 1])
    reports = tf.concat([coalition_reports, other_reports], axis=0)
    reports = tf.clip_by_value(reports, EPSILON, 1 - EPSILON)
    preferences = tf.reshape(preferences, [N_COALITION, M])
    weights = tf.fill([N_TOTAL], 1/N_TOTAL)
    beliefs = tf.linalg.matvec(reports, weights, transpose_a=True)
    # sigmoid instead of step function for differentiability
    allocation = sigmoid(beliefs, THRESHOLD, 5)
    # allocation = tf.cast(tf.greater(beliefs, THRESHOLD), dtype=tf.float32)
    desirability_utilities = preferences * \
        tf.tile(tf.reshape(allocation, [1, M]), [N_COALITION, 1])
    return -1 * tf.math.reduce_sum(desirability_utilities)


def desirability_loss(y_true, y_pred):
    # y_true is ignored:
    # utility from inherent preference of borrower based on allocation of loan, not that borrower's repayment
    # y_pred is the learned reports concatenated with the inputs (how much each recommender cares about each borrower)
    reports, preferences = tf.split(y_pred, 2, axis=0)
    result = tf.map_fn(
        lambda x: calculate_loss_in_desired_borrowers(x[0], x[1]), (reports, preferences), fn_output_signature=tf.float32)
    return tf.math.reduce_sum(result)


def mixed_loss(desirability_importance=0.5):

    def desirability_and_profit_loss(y_true, y_pred):
        reports, _ = tf.split(y_pred, 2, axis=0)
        return desirability_importance * desirability_loss(y_true, y_pred) + \
            (1-desirability_importance) * profit_loss(y_true,
                                                      reports)  # need to remove preferences for profit calc

    return desirability_and_profit_loss


def profit_reports() -> None:
    # here we are just maximizing profit from the mechanism, resulting in accurate reports
    # X doesn't do anything, it's just stochastic noise for the network to run
    X = np.random.random((N_TEST_CASES, N_COALITION, M))
    # X = np.full((N_TEST_CASES, N * M), 0.5)
    # y is the binary outcome of repayment (if allocated to that borrower)
    # we sample y according to the true probabilities so the probability is in the training
    # data, rather than the loss function (calling a random function is not differentiable)
    y = np.random.binomial(1, PROBS, (N_TEST_CASES, M)).astype(np.float32)

    inputs = Input(shape=(N_COALITION, M), dtype=tf.float32)
    layer = Flatten()(inputs)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M, activation='sigmoid')(layer)
    outputs = Reshape((N_COALITION, M))(layer)

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    model.compile(loss=profit_loss,
                  optimizer=Adam(amsgrad=True))

    # plot_model(model, 'model.png', show_shapes=True)

    history = model.fit(X, y, validation_split=0.2,
                        epochs=30, batch_size=BATCH_SIZE, verbose=0)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Model loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.show()

    predictions = model.predict(np.full((1, N_COALITION, M), 0.5))[0]
    print('mean: ' + str(np.mean(predictions, axis=0)))
    print('std: ' + str(np.std(predictions, axis=0)))
    print('')


def desire_borrowers() -> None:
    # in this case the quantity that we are maximizing is actually the probabilities
    # that the desired people get loans
    # here, x_{i,q} represents how much that recommender i wants q to get a loan.

    # when the coalition all agrees on one borrower
    # indices = np.random.randint(0, M, N_TEST_CASES)
    # X = np.array([np.tile(np.reshape([int(j == index) for j in range(M)], (1, M)), (N_COALITION, M))
    #               for _, index in enumerate(indices)], dtype=np.float32)

    X = np.random.randint(0, 2, (N_TEST_CASES, N_COALITION, M))

    # y is ignored
    y = np.zeros((N_TEST_CASES, 1)).astype(np.float32)

    inputs = Input(shape=(N_COALITION, M), dtype=tf.float32)
    layer = Flatten()(inputs)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M, activation='sigmoid')(layer)
    # we have to concatenate the input biases in the final layer so they can appear in the loss function
    # but the input biases never actually change
    outputs = Concatenate(axis=0)([Reshape((1, N_COALITION, M))(layer),
                                   Reshape((1, N_COALITION, M))(inputs)])

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    model.compile(loss=desirability_loss,
                  optimizer=Adam(amsgrad=True))
    history = model.fit(X, y, validation_split=0.2,
                        epochs=30, batch_size=BATCH_SIZE, verbose=0)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Model loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.show()

    # iterate over borrowers that recommenders care about
    tests = []
    for index in range(M):
        test = np.zeros((1, M))
        test[0, index] = 1
        test = np.tile(test, [N_COALITION, 1])
        tests.append(test)
    predictions = model.predict(np.array(tests))
    predictions, _ = np.split(predictions, 2, axis=0)
    predictions = np.array([prediction[0] for prediction in predictions])
    if PRINT_COMMENTS:
        for prediction in predictions:
            print('mean: ' + str(np.mean(prediction, axis=0)))
            print('std: ' + str(np.std(prediction, axis=0)))
            print('')
    return predictions


def desire_borrowers_and_profit(alpha=0.5, testname=None):
    # in this case recommenders care about both profits and helping their desired borrower get a loan

    # here, x_{i,q} represents how much that recommender i wants q to get a loan.
    indices = np.random.randint(0, M, N_TEST_CASES)
    X = np.array([np.tile(np.reshape([float(j == index) for j in range(M)], (1, M)), (N_COALITION, 1))
                  for _, index in enumerate(indices)])

    # X = np.random.randint(0, 2, (N_TEST_CASES, N_COALITION * M))

    # y is the binary outcome of repayment (if allocated to that borrower)
    # we sample y according to the true probabilities so the probability is in the training
    # data, rather than the loss function (calling a random function is not differentiable)
    y = np.random.binomial(1, PROBS, (N_TEST_CASES, M)).astype(np.float32)

    inputs = Input(shape=(N_COALITION, M), dtype=tf.float32)
    layer = Flatten()(inputs)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M, activation='sigmoid')(layer)
    # we have to concatenate the input biases in the final layer so they can appear in the loss function
    # but the input biases never actually change
    outputs = Concatenate(axis=0)([Reshape((1, N_COALITION, M))(layer),
                                   Reshape((1, N_COALITION, M))(inputs)])

    model = Model(inputs=inputs, outputs=outputs, name="collusion_model")

    # the mixed loss coefficient is what percent the recommenders care about their desired borrower
    # as compared to profit
    model.compile(loss=mixed_loss(alpha),
                  optimizer=Adam(amsgrad=True))
    history = model.fit(X, y, validation_split=0.2,
                        epochs=50, batch_size=BATCH_SIZE, verbose=0)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Model loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(f'results/{testname}_alpha_{np.round(alpha, 2)}.png',
                bbox_inches='tight')
    plt.close()
    # plt.show()

    # iterate over borrowers that recommenders care about
    tests = []
    for index in range(M):
        test = np.zeros((1, M))
        test[0, index] = 1
        test = np.tile(test, [N_COALITION, 1])
        tests.append(test)
    predictions = model.predict(np.array(tests))
    predictions, _ = np.split(predictions, 2, axis=0)
    predictions = np.array([prediction[0] for prediction in predictions])
    if PRINT_COMMENTS:
        for prediction in predictions:
            print('alpha: ' + str(alpha))
            print(prediction)
            print('mean: ' + str(np.mean(prediction, axis=0)))
            print('std: ' + str(np.std(prediction, axis=0)))
            print('')
    return predictions


def test_alpha():
    predictions = []
    alphas = np.linspace(0, 1, num=11)
    for alpha in alphas:
        print(alpha)
        prediction = desire_borrowers_and_profit(alpha, 'test_alpha')
        predictions.append(prediction)
    predictions = np.array(predictions)
    with open('results/nn_alpha_predictions.npy', 'wb') as f:
        np.save(f, predictions)


def test_alpha_post_process_aggregate():
    alphas = np.linspace(0, 1, num=11)
    with open('saved_results/nn_alpha_predictions.npy', 'rb') as f:
        predictions = np.load(f)
        deviations = []  # difference between true probs and actual reports
        deviations_stdev = []
        deviations_collusive_report = []
        deviations_collusive_report_stdev = []
        deviations_non_collusive_report = []
        deviations_non_collusive_report_stdev = []
        diff_in_collusive_report = []
        diff_in_collusive_report_stdev = []
        for prediction in predictions:
            dev = [pred - np.tile(np.reshape(PROBS, [1, M]),
                                  [N_COALITION, 1]) for pred in prediction]
            deviations.append(np.mean(dev))
            deviations_stdev.append(np.std(dev))
            collusive_dev = []
            non_collusive_dev = []
            for i, dev in enumerate(dev):
                collusive_dev.append(dev[:, i])
                non_collusive_dev.append(np.delete(dev, i, 1))
            deviations_collusive_report.append(np.mean(collusive_dev))
            deviations_collusive_report_stdev.append(np.std(collusive_dev))
            deviations_non_collusive_report.append(np.mean(non_collusive_dev))
            deviations_non_collusive_report_stdev.append(
                np.std(non_collusive_dev))
            collusive_dev_mn = np.array([np.mean(x) for x in collusive_dev])
            non_collusive_dev_mn = np.array(
                [np.mean(x) for x in non_collusive_dev])
            diff_in_collusive_report.append(
                np.mean(collusive_dev_mn - non_collusive_dev_mn))
            diff_in_collusive_report_stdev.append(
                np.std(collusive_dev_mn - non_collusive_dev_mn))

        save_fig('results/collusion_alpha.png', 'Average Deviation from True Reports',
                 'alpha', 'Average deviation', alphas, deviations, deviations_stdev)

        save_fig('results/collusion_desired_alpha.png', 'Average Deviation from True Reports on Desired Borrowers',
                 'alpha', 'Average deviation', alphas, deviations_collusive_report, deviations_collusive_report_stdev)

        save_fig('results/collusion_non_desired_alpha.png', 'Average Deviation from True Reports on Non-Desired Borrowers',
                 'alpha', 'Average deviation', alphas, deviations_non_collusive_report, deviations_non_collusive_report_stdev)

        save_fig('results/collusion_difference_alpha.png', 'Average Difference in Deviation from True Reports between Desired and Non-Desired Borrowers',
                 'alpha', 'Average deviation', alphas, diff_in_collusive_report, diff_in_collusive_report_stdev)


def test_alpha_post_process_per_borrower():
    alphas = np.linspace(0, 1, num=11)
    with open('saved_results/nn_alpha_predictions.npy', 'rb') as f:
        predictions = np.load(f)
        deviations = []  # difference between true probs and actual reports
        deviations_stdev = []
        deviations_collusive_report = []
        deviations_non_collusive_report = []
        diff_in_collusive_report = []
        for prediction in predictions:
            dev = [pred - np.tile(np.reshape(PROBS, [1, M]),
                                  [N_COALITION, 1]) for pred in prediction]
            deviations.append([np.mean(x) for x in dev])
            deviations_stdev.append([np.std(x) for x in dev])
            collusive_dev = []
            non_collusive_dev = []
            for i, dev in enumerate(dev):
                collusive_dev.append(dev[:, i])
                non_collusive_dev.append(np.delete(dev, i, 1))
            deviations_collusive_report.append(
                [np.mean(x) for x in collusive_dev])
            deviations_non_collusive_report.append(
                [np.mean(x) for x in non_collusive_dev])
            diff_in_collusive_report.append(
                [np.mean(x) - np.mean(y) for x, y in zip(collusive_dev, non_collusive_dev)])

        save_fig_multiy('results/collusion_alpha_borrower.png', 'Average Deviation from True Reports',
                        'alpha', 'Average deviation', alphas, deviations)

        save_fig_multiy('results/collusion_desired_alpha_borrower.png', 'Average Deviation from True Reports on Desired Borrowers',
                        'alpha', 'Average deviation', alphas, deviations_collusive_report)

        save_fig_multiy('results/collusion_non_desired_alpha_borrower.png', 'Average Deviation from True Reports on Non-Desired Borrowers',
                        'alpha', 'Average deviation', alphas, deviations_non_collusive_report)

        save_fig_multiy('results/collusion_difference_alpha_borrower.png', 'Average Difference in Deviation from True Reports between Desired and Non-Desired Borrowers',
                        'alpha', 'Average deviation', alphas, diff_in_collusive_report)


def test_coalition_size():
    global N_COALITION
    coalition_sizes = np.array([x + 1 for x in range(N_TOTAL)])
    predictions = []
    for size in coalition_sizes:
        print(size)
        N_COALITION = size
        prediction = desire_borrowers_and_profit(0.4, 'test_coalition_size')
        predictions.append(prediction)
    N_COALITION = 3
    print(predictions)
    with open('results/nn_size_predictions.npz', 'wb') as f:
        # unpack predictions
        np.savez(f, *predictions)


def test_coalition_size_post_process():
    coalition_sizes = np.array([x + 1 for x in range(N_TOTAL)])
    with open('results/nn_size_predictions.npz', 'rb') as f:
        npzfile = np.load(f)
        predictions = []
        for file in npzfile.files:
            predictions.append(npzfile[file])
        deviations = []  # difference between true probs and actual reports
        deviations_collusive_report = []
        deviations_non_collusive_report = []
        diff_in_collusive_report = []
        for i, prediction in enumerate(predictions):
            dev = [pred - np.tile(np.reshape(PROBS, [1, M]),
                                  [i+1, 1]) for pred in prediction]
            # coalition size is i+1
            deviations.append([np.mean(x) for x in dev])
            collusive_dev = []
            non_collusive_dev = []
            for i, dev in enumerate(dev):
                collusive_dev.append(dev[:, i])
                non_collusive_dev.append(np.delete(dev, i, 1))
            deviations_collusive_report.append(
                [np.mean(x) for x in collusive_dev])
            deviations_non_collusive_report.append(
                [np.mean(x) for x in non_collusive_dev])
            diff_in_collusive_report.append(
                [np.mean(x) - np.mean(y) for x, y in zip(collusive_dev, non_collusive_dev)])

        save_fig_multiy('results/collusion_size.png', 'Average Deviation from True Reports',
                        'Coalition size', 'Average deviation', coalition_sizes, deviations)

        save_fig_multiy('results/collusion_desired_size.png', 'Average Deviation from True Reports on Desired Borrowers',
                        'Coalition size', 'Average deviation', coalition_sizes, deviations_collusive_report)

        save_fig_multiy('results/collusion_non_desired_size.png', 'Average Deviation from True Reports on Non-Desired Borrowers',
                        'Coalition size', 'Average deviation', coalition_sizes, deviations_non_collusive_report)

        save_fig_multiy('results/collusion_difference_size.png', 'Average Difference in Deviation from True Reports between Desired and Non-Desired Borrowers',
                        'Coalition size', 'Average deviation', coalition_sizes, diff_in_collusive_report)


def save_fig(filename, title, xlabel, ylabel, x, y, err=None):
    if err:
        plt.errorbar(x, y, err, ecolor='black',
                     elinewidth=0.5, capsize=3, capthick=0.5)
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_fig_multiy(filename, title, xlabel, ylabel, x, y, err=None, serieslabels=[i+1 for i in range(M)], legendtitle="Preferred borrower"):
    x = np.array(x)
    y = np.hsplit(np.array(y), M)
    if err:
        err = np.hsplit(np.array(err), M)
        for yseries, errseries in zip(y, err):
            plt.errorbar(x, yseries.ravel(), errseries.ravel(), ecolor='black',
                         elinewidth=0.5, capsize=3, capthick=0.5)
    else:
        for yseries in y:
            plt.plot(x, yseries.ravel())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(serieslabels, title=legendtitle, loc='upper left')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main() -> None:
    global PRINT_COMMENTS
    PRINT_COMMENTS = True
    # profit_reports()
    # desire_borrowers()
    # desire_borrowers_and_profit(0.8)
    # desire_borrowers_and_profit(0.7)
    # desire_borrowers_and_profit(0.5)
    # desire_borrowers_and_profit(0.3)

    # test_alpha()
    # test_alpha_post_process_aggregate()
    # test_alpha_post_process_per_borrower()

    test_coalition_size()
    test_coalition_size_post_process()


if __name__ == '__main__':
    main()
