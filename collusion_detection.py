# collusion_detection.py
# please use Python >=3.9

from enum import Enum, auto
# from keras.models import Sequential
# from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from statistics import NormalDist
from typing import Optional, Union


class ReportStrategy(Enum):
    TRUE_BELIEFS = auto()
    COLLUSIVE_UP = auto()       # upvote
    COLLSUIVE_ADV = auto()      #


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

    def gen(self, type: ReportStrategy) -> tuple[npt.NDArray, npt.NDArray]:
        '''returns outcomes and reports'''
        if type == ReportStrategy.TRUE_BELIEFS:
            return self.outcomes, self.true_beliefs
        elif type == ReportStrategy.SINGLE_COLLUSIVE:
            return self.outcomes, np.clip(self.true_beliefs + self.collusive_bias,
                                          0, 1)
        else:
            assert False, 'gen_reports(): invalid report strategy'


class LendingDataGenerator:
    def __init__(self, alpha: int = 6, beta: int = 4) -> None:
        self.alpha = alpha
        self.beta = beta

    def draw_sample(self,
                    size: Optional[Union[int, tuple[int]]] = None
                    ) -> Union[float, npt.NDArray]:
        return np.random.beta(self.alpha, self.beta, size)


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
        n = 30
        true_scores = []
        collusive_scores = []

        for _ in range(n):
            outcomes, predictions = SingleRecommenderDataGenerator(collusive_bias=collusive_bias).gen(
                ReportStrategy.TRUE_BELIEFS)
            true_scores.append(brier_score(outcomes, predictions))

            outcomes, predictions = SingleRecommenderDataGenerator(collusive_bias=collusive_bias).gen(
                ReportStrategy.SINGLE_COLLUSIVE)
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


def brier_score(outcomes: npt.ArrayLike, predictions: npt.ArrayLike) -> float:
    return mean_squared_error(outcomes, predictions)


def main() -> None:
    test_brier()


if __name__ == '__main__':
    main()
