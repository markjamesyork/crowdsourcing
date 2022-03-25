# collusion_detection.py
# please use Python >=3.9

from enum import Enum, auto
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats as scipystats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler
from statistics import NormalDist

from coalition_winkler import mixed_loss


auc = AUC()


class ReportStrategy(Enum):
    TRUE_BELIEFS = auto()
    COLLUSIVE_UP = auto()       # upvote
    COLLUSIVE_ADV = auto()      # upvote some, downvote others
    COLLUSIVE_FRAC = auto()


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
                 collusive_bias: float = 0.1,
                 coalition_size: int = 1,
                 ) -> None:
        self.alpha = alpha
        self.beta = beta

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = m
        self.collusive_bias = collusive_bias
        self.coalition_size = coalition_size

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
            collusive_js = random.randint(0, self.m - 1)

            reports = np.copy(self.true_beliefs)
            reports[collusive_i, :] = np.clip(reports[collusive_i, :] +
                                              np.array([self.collusive_bias
                                                        if j in collusive_js
                                                        else -self.collusive_bias
                                                        for j in range(self.m)]),
                                              0, 1)
            return self.outcomes, reports, collusive_i
        elif type == ReportStrategy.COLLUSIVE_FRAC:
            collusive_j = random.randint(0, self.m - 1)
            recommender_arr = [i for i in range(self.n)]
            np.random.shuffle(recommender_arr)
            reports = np.copy(self.true_beliefs)
            for index in range(self.coalition_size):
                i = recommender_arr[index]
                reports[i, :] = np.clip(reports[i, :] +
                                        np.array([self.collusive_bias
                                                  if j == collusive_j
                                                  else -self.collusive_bias
                                                  for j in range(self.m)]),
                                        0, 1)
            return self.outcomes, reports, recommender_arr[:self.coalition_size]
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
    n, m = 10, 10
    data = []
    output = []
    for _ in range(N_ITERS):
        _, reports, collusive_is = RecommenderDataGenerator(n=n, m=m, coalition_size=2).gen(
            ReportStrategy.COLLUSIVE_FRAC)
        for i in range(n):
            # remove effect of recommender i
            avg_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
            sd_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
            probs = []
            for j in range(len(sd_reports)):
                prob = prob_not_as_extreme(
                    reports[i, j], avg_reports[j], sd_reports[j])
                probs.append(prob)
            data.append(np.array(probs))
            output.append(i in collusive_is)

    X = np.array(data)
    y = np.array(output)

    X = np.array(MinMaxScaler().fit_transform(X))

    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=m))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[auc])
    history = model.fit(X, y, validation_split=0.2,
                        epochs=50, batch_size=32, verbose=0)

    model.save('detection_knowledge')
    # for layer in model.layers:
    #     weights = layer.get_weights()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Binary cross-entropy loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def test_knowledge_coalition_size() -> None:
    N_ITERS = 50
    n, m = 10, 10

    model = load_model('detection_knowledge')
    print(model.metrics_names)

    evaluations = []
    for collusive_n in range(1, n+1):
        data = []
        output = []
        for _ in range(N_ITERS):
            _, reports, collusive_is = RecommenderDataGenerator(n=n, m=m, coalition_size=collusive_n).gen(
                ReportStrategy.COLLUSIVE_FRAC)
            for i in range(n):
                # remove effect of recommender i
                avg_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
                sd_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
                probs = []
                for j in range(len(sd_reports)):
                    prob = prob_not_as_extreme(
                        reports[i, j], avg_reports[j], sd_reports[j])
                    probs.append(prob)
                data.append(np.array(probs))
                output.append(i in collusive_is)

        X = np.array(data)
        y = np.array(output)

        X = np.array(MinMaxScaler().fit_transform(X))

        evaluation = model.evaluate(X, y, verbose=0)
        evaluations.append(evaluation)

    evaluations = np.array(evaluations)
    print(evaluations)
    with open('test_knowledge_coalition_size.npy', 'wb') as f:
        np.save(f, evaluations)

    aucs = np.array(evaluations[:, 1].flatten())
    plt.plot([i+1 for i in range(n)], aucs)
    plt.ylabel('AUC')
    plt.xlabel('Coalition size')
    plt.show()


def test_creditworthiness() -> None:
    N_ITERS = 100
    n, m = 10, 10
    data = []
    output = []
    for _ in range(N_ITERS):
        _, reports, collusive_is = RecommenderDataGenerator(n=n, m=m, coalition_size=2).gen(
            ReportStrategy.COLLUSIVE_FRAC)
        for i in range(n):
            # remove effect of recommender i
            avg_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
            params = scipystats.beta.fit(avg_reports, floc=0, fscale=1)
            ksstat, pval = scipystats.kstest(reports[i], 'beta', args=params)
            data.append([ksstat, pval])
            output.append(i in collusive_is)

    X = np.array(data)
    y = np.array(output)

    X = np.array(MinMaxScaler().fit_transform(X))

    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(amsgrad=True),
                  metrics=[auc])
    history = model.fit(X, y, validation_split=0.2,
                        epochs=50, batch_size=32, verbose=0)

    model.save('detection_creditworthiness')
    # for layer in model.layers:
    #     weights = layer.get_weights()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Binary cross-entropy loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def test_creditworthiness_coalition_size() -> None:
    N_ITERS = 50
    n, m = 10, 10

    model = load_model('detection_creditworthiness')
    print(model.metrics_names)

    evaluations = []
    for collusive_n in range(1, n+1):
        data = []
        output = []
        for _ in range(N_ITERS):
            _, reports, collusive_is = RecommenderDataGenerator(n=n, m=m, coalition_size=collusive_n).gen(
                ReportStrategy.COLLUSIVE_FRAC)
            for i in range(n):
                # remove effect of recommender i
                avg_reports = np.mean(np.delete(reports, i, axis=0), axis=0)
                params = scipystats.beta.fit(avg_reports, floc=0, fscale=1)
                ksstat, pval = scipystats.kstest(
                    reports[i], 'beta', args=params)
                data.append([ksstat, pval])
                output.append(i in collusive_is)

        X = np.array(data)
        y = np.array(output)

        X = np.array(MinMaxScaler().fit_transform(X))

        evaluation = model.evaluate(X, y, verbose=0)
        evaluations.append(evaluation)

    evaluations = np.array(evaluations)
    print(evaluations)
    with open('test_creditworthiness_coalition_size.npy', 'wb') as f:
        np.save(f, evaluations)

    aucs = np.array(evaluations[:, 1].flatten())
    plt.plot([i+1 for i in range(n)], aucs)
    plt.ylabel('AUC')
    plt.xlabel('Coalition size')
    plt.show()


def test_learned_collusion():
    model = load_model('coalition_alpha_08', custom_objects={
                       'desirability_and_profit_loss': mixed_loss(0.8)})
    N_TEST_CASES = 5000
    N_COALITION = 3
    N_TOTAL = 6
    M = 4
    PROBS = [0.43, 0.62, 0.70, 0.76]
    tests = np.random.randint(0, 2, (N_TEST_CASES, N_COALITION, M))
    predictions = model.predict(np.array(tests))
    predictions, _ = np.split(predictions, 2, axis=0)
    predictions = np.array([prediction[0] for prediction in predictions])

    true_reports = np.array([np.clip(np.tile(np.reshape(PROBS, (1, M)), (N_TOTAL - N_COALITION, 1)) +
                                     np.random.normal(0, 0.1, (N_TOTAL - N_COALITION, M)), 0, 1) for _ in range(N_TEST_CASES)])

    X = []
    y = []
    for collusive_report, true_report in zip(predictions, true_reports):
        rand_indices = get_indices(N_TOTAL, N_COALITION)
        for report, i in zip(collusive_report, rand_indices):
            true_report = np.insert(true_report, i, report, axis=0)
        yi = [float(i in rand_indices) for i in range(N_TOTAL)]
        X.append(np.reshape(true_report, N_TOTAL * M))
        y.append(yi)
    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(Dense(N_TOTAL * M, activation='relu', input_dim=N_TOTAL * M))
    model.add(Dense(N_TOTAL * M/2, activation='relu'))
    model.add(Dense(N_TOTAL, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(amsgrad=True),
                  metrics=[AUC()])
    history = model.fit(X, y, validation_split=0.2,
                        epochs=200, batch_size=32, verbose=0)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Binary crossentropy loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def get_indices(n, k):
    assert k < n
    result = []
    indices = [i for i in range(n)]
    for _ in range(k):
        selected = random.choice(indices)
        indices.remove(selected)
        result.append(selected)
    return sorted(result)


def main() -> None:
    # test_brier()

    # test_knowledge()
    # test_knowledge_coalition_size()

    test_creditworthiness()
    test_creditworthiness_coalition_size()

    # test_learned_collusion()


if __name__ == '__main__':
    main()
