# collusion_detection.py
# please use Python >=3.9

from enum import Enum, auto
from keras.models import Sequential
from keras.layers import Dense
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
from typing import Optional, Union


class ReportStrategy(Enum):
    TRUE_BELIEFS = auto()
    COLLUSIVE_UP = auto()       # upvote
    COLLUSIVE_ADV = auto()      # upvote some, downvote others


class SingleRecommenderDataGenerator:
    def __init__(self,
                 alpha: int = 6,
                 beta: int = 4,
                 n_reports: int = 50,
                 belief_bias: float = 0,
                 belief_sd: float = 0.05,
                 collusive_bias: float = 0.1
                 ) -> None:
        assert isinstance(alpha, int)
        assert isinstance(beta, int)
        assert isinstance(n_reports, int)
        assert n_reports > 0

        self.alpha = alpha
        self.beta = beta
        self.n_reports = n_reports
        self.collusive_bias = collusive_bias

        self.true_probs = np.random.beta(alpha, beta, n_reports)
        self.true_beliefs = np.clip(self.true_probs +
                                    np.random.normal(
                                        belief_bias, belief_sd, n_reports),
                                    0, 1)
        self.outcomes = np.random.binomial(1, self.true_probs)

    def gen(self, type: ReportStrategy):
        '''returns outcomes and reports'''
        if type == ReportStrategy.TRUE_BELIEFS:
            return self.outcomes, self.true_beliefs
        elif type == ReportStrategy.COLLUSIVE_UP:
            return self.outcomes, np.clip(self.true_beliefs + self.collusive_bias,
                                          0, 1)
        else:
            assert False, 'gen_reports(): invalid report strategy'


class RecommenderDataGenerator:
    def __init__(self,
                 alpha: int = 6,
                 beta: int = 4,
                 n: int = 15,
                 m: int = 15,
                 belief_bias: float = 0,
                 belief_sd: float = 0.05,
                 collusive_bias: float = 0.15,
                 collusive_frac: float = 0.1,
                 ) -> None:
        self.alpha = alpha
        self.beta = beta

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = m
        self.collusive_bias = collusive_bias
        self.collusive_frac = collusive_frac

        self.true_probs = np.random.beta(alpha, beta, m)
        self.true_beliefs = np.clip(np.tile(self.true_probs.transpose(), (self.n, 1)) +
                                    np.random.normal(
                                        belief_bias, belief_sd, (n, m)),
                                    0, 1)
        self.outcomes = np.random.binomial(1, self.true_probs)

    def gen(self, type: ReportStrategy):
        '''returns outcomes and reports'''
        if type == ReportStrategy.TRUE_BELIEFS:
            return self.outcomes, self.true_beliefs
        elif type == ReportStrategy.COLLUSIVE_ADV:
            assert self.n > 1
            assert self.m > 2
            collusive_i = random.randint(0, self.n - 1)
            collusive_js = [random.randint(0, self.m - 1) for _ in range(5)]

            reports = np.copy(self.true_beliefs)
            reports[collusive_i, :] = np.clip(reports[collusive_i, :] +
                                              np.array([self.collusive_bias
                                                        if j in collusive_js
                                                        else -1 * self.collusive_bias
                                                        for j in range(self.m)]),
                                              0, 1)
            return self.outcomes, reports, collusive_i
        else:
            assert False, 'gen_reports(): invalid report strategy'


class SingleClassificationModel:
    '''Classifying collusive behaviors, looking at a single recommender only'''

    def __init__(self, guesses, classification):
        pass


def prob_not_as_extreme(x: float, mean: float, sd: float) -> float:
    '''prob. a sample from Norm(mean, sd) is not as extreme as x'''
    distribution = NormalDist(mu=mean, sigma=sd)
    return 1 - 2 * distribution.cdf(mean - abs(mean - x))


def generate_true_predictions():
    pass


def test_brier() -> None:
    for collusive_bias in [0.05, 0.1, 0.2, 0.3]:
        n = 15
        true_scores = []
        collusive_scores = []

        for _ in range(n):
            outcomes, predictions = SingleRecommenderDataGenerator(collusive_bias=collusive_bias).gen(
                ReportStrategy.TRUE_BELIEFS)
            true_scores.append(brier_score(outcomes, predictions))

            outcomes, predictions = SingleRecommenderDataGenerator(collusive_bias=collusive_bias).gen(
                ReportStrategy.COLLUSIVE_UP)
            collusive_scores.append(brier_score(outcomes, predictions))

        plt.scatter(np.full(n, 0.4), true_scores, label='true reports')
        plt.scatter(np.full(n, 0.6), collusive_scores,
                    label=f'collusive bias={collusive_bias}')
        plt.ylabel('Brier score')
        plt.xlim(left=0, right=1)
        plt.legend(loc="upper left")
        plt.title('Brier scores for true vs. collusive reports')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.show()

        # generator = SingleRecommenderDataGenerator()
        # x = np.random.binomial()


def brier_score(outcomes, predictions) -> float:
    return mean_squared_error(outcomes, predictions)


def test_knowledge() -> None:
    N_ITERS = 100
    n, m = 10, 15
    data = []
    output = []
    for _ in range(N_ITERS):
        _, reports, collusive_i = RecommenderDataGenerator(n=n, m=m).gen(
            ReportStrategy.COLLUSIVE_ADV)
        included_non_colluder = False
        for i in range(n):
            # remove effect of recommender i
            avg_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
            sd_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
            probs = []
            for j in range(len(sd_reports)):
                prob = prob_not_as_extreme(
                    reports[i, j], avg_reports[j], sd_reports[j])
                probs.append(prob)
            if i == collusive_i:
                data.append(np.array(probs))
                output.append(i == collusive_i)
            elif not included_non_colluder:
                included_non_colluder = True
                data.append(np.array(probs))
                output.append(i == collusive_i)

    X = np.array(data)
    y = np.array(output)

    X = np.array(MinMaxScaler().fit_transform(X))

    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=m))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.2,
                        epochs=100, batch_size=10, verbose=0)
    # for layer in model.layers:
    #     weights = layer.get_weights()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    print(sum(y) / len(y))


def test_creditworthiness() -> None:
    N_ITERS = 100
    n, m = 10, 15
    data = []
    output = []
    for _ in range(N_ITERS):
        _, reports, collusive_i = RecommenderDataGenerator(n=n, m=m).gen(
            ReportStrategy.COLLUSIVE_ADV)
        included_non_colluder = False
        for i in range(n):
            # remove effect of recommender i
            avg_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
            params = scipystats.beta.fit(avg_reports, floc=0, fscale=1)
            ksstat, pval = scipystats.kstest(reports[i], 'beta', args=params)
            if i == collusive_i:
                data.append([ksstat, pval])
                output.append(i == collusive_i)
            elif not included_non_colluder:
                included_non_colluder = True
                data.append([ksstat, pval])
                output.append(i == collusive_i)

    X = np.array(data)
    y = np.array(output)

    X = np.array(MinMaxScaler().fit_transform(X))

    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.2,
                        epochs=100, batch_size=10, verbose=0)
    # for layer in model.layers:
    #     weights = layer.get_weights()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    print(sum(y) / len(y))


def test_calibration():
    pass


def test_optimal_collusion():
    pass


def main() -> None:
    # test_brier()
    test_knowledge()
    # test_creditworthiness()


if __name__ == '__main__':
    main()
