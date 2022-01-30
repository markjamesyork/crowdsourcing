# agents.py

from enum import Enum, auto
import numpy as np
import numpy.typing as npt
import numpy.testing as testing
import pulp


class BeliefNoise(Enum):
    ZERO = auto()
    GAUSSIAN = auto()


class ReportStrategy(Enum):
    TRUE = auto()
    GAUSSIAN = auto()


class ElicitationStrategy(Enum):
    WINKLER = auto()
    VCG = auto()


class Model:
    '''Simulates borrowers-recommenders-lender system'''

    @property
    def EPSILON(self) -> float:
        return 0.01

    def __init__(self,
                 n: int,    # number of recommenders
                 m: int,    # number of borrowers
                 true_probabilities: npt.ArrayLike = None,  # repayment probabilities
                 true_beliefs: npt.ArrayLike = None,
                 threshold: float = 0.5,                    # lender's threshold
                 weights: npt.ArrayLike = None,             # weights given to recommenders
                 liquidity: float = None,
                 ) -> None:
        assert n > 0, 'init(): n must be positive'
        assert m > 0, 'init(): m must be positive'

        # borrowers
        self.m = np.int32(m)
        self.true_probabilities = np.array(nullish(true_probabilities, np.random.rand(m)),
                                           dtype=np.float64)
        self.outcomes = np.zeros(m, np.int32)

        # recommenders
        self.n = np.int32(n)
        # recommenders have same true beliefs by default
        self.true_beliefs = np.array(nullish(true_beliefs, np.tile(self.true_probabilities.transpose(), (self.n, 1))),
                                     dtype=np.float64)

        self.reports = np.zeros((n, m))
        self.immediate_payments = np.zeros(n)
        self.outcome_payments = np.zeros((n, m))

        # lender
        self.threshold = np.float64(threshold)
        self.weights = np.array(nullish(weights, np.full(n, 1/n)),
                                dtype=np.float64)
        self.liquidity = np.int32(nullish(liquidity, m))
        self.allocation = np.zeros(m, np.int32)

        # flags to ensure proper usage
        self.REPORT_STRATEGY = None
        self.ELICITATION_STRATEGY = None

        # normalization checks
        assert np.logical_and(self.true_probabilities >= 0,
                              self.true_probabilities <= 1).all(), 'init(): True probabilities must be in [0, 1]'
        assert np.logical_and(self.true_beliefs >= 0,
                              self.true_beliefs <= 1).all(), 'init(): True beliefs must be in [0, 1]'
        assert 0 <= self.threshold <= 1, 'init(): Threshold must be in [0, 1]'
        assert np.logical_and(self.weights >= 0,
                              self.weights <= 1).all(), 'init(): Weights must be in [0, 1]'
        testing.assert_almost_equal(
            np.sum(self.weights), 1), 'init(): Recommender weights must be normalized'
        assert 1 <= self.liquidity <= self.m, 'init(): Liquidity must be in {1, ..., m}'

        # dimensionality checks
        assert self.true_probabilities.shape == (
            self.m, ), 'init(): Dimension of true probabilities must be (n)'
        assert self.true_beliefs.shape == (
            self.n, self.m), 'init(): Dimension of true beliefs must be (n x m)'
        assert self.reports.shape == (
            self.n, self.m), 'init(): Dimension of recommender reports must be (n x m)'
        assert self.weights.shape == (
            self.n, ), 'init(): Dimensions of recommender weights must be (n)'
        assert self.allocation.shape == (
            self.m, ), 'init(): Dimensions of allocation must be (m)'
        assert self.outcomes.shape == (
            self.m, ), 'init(): Dimensions of outcomes must be (m)'
        assert self.immediate_payments.shape == (
            self.n, ), 'init(): Dimension of immediate payments must be (n)'
        assert self.outcome_payments.shape == (
            self.n, self.m), 'init(): Dimension of outcome payments must be (n x m)'

    def add_beliefs_noise(self, type: BeliefNoise, param: float = 0.05) -> None:
        if type == BeliefNoise.ZERO:
            pass
        elif type == BeliefNoise.GAUSSIAN:
            self.true_beliefs += np.random.normal(0, param, (self.n, self.m))
        else:
            assert False, 'add_beliefs_noise(): Belief noise type invalid'
        self.true_beliefs = truncate(self.true_beliefs)

    def make_reports(self, type: ReportStrategy, param: float = 0.05) -> None:
        self.REPORT_STRATEGY = type
        if type == ReportStrategy.TRUE:
            self.reports = self.true_beliefs
        elif type == ReportStrategy.GAUSSIAN:
            self.reports = self.true_beliefs + \
                np.random.normal(0, param, (self.n, self.m))
        else:
            assert False, 'make_reports(): Report type invalid'
        self.reports = truncate(self.reports, min=self.EPSILON)

    def elicit(self, type: ElicitationStrategy) -> None:
        assert self.REPORT_STRATEGY, 'must call make_reports() before calling elicit()'
        assert not self.ELICITATION_STRATEGY, 'cannot elicit() twice'
        self.ELICITATION_STRATEGY = type
        if type == ElicitationStrategy.WINKLER:
            self._elicit_winkler()
        elif type == ElicitationStrategy.VCG:
            self._elicit_vcg()
        else:
            assert False, 'elicit(): Elicitation strategy invalid'

    def _get_score_winkler(self, i: int, q: int) -> np.float64:
        '''util for _elicit_winkler()'''
        report = self.reports[i, q]
        outcome = self.outcomes[q]
        min_report = truncate(1/self.weights[i] * (self.threshold - (self._get_linear_aggregator(q) - self.weights[i] * report)),
                              min=self.EPSILON)

        assert outcome == 0 or outcome == 1, '_get_score_winkler(): Outcome must be binary'

        if report <= min_report:
            return 0.0
        elif outcome:
            return (np.log(report) - np.log(min_report)) / (-1 * np.log(min_report))
        else:
            return (np.log(1 - report) - np.log(1 - min_report)) / (-1 * np.log(min_report))

    def _get_linear_aggregator(self, q: int) -> np.float64:
        '''util for _get_linear_aggregator()'''
        return np.dot(self.weights, self.reports[:, q])

    def _elicit_winkler(self) -> None:
        for q in range(self.m):
            belief = self._get_linear_aggregator(q)
            if belief > self.threshold:
                self.allocation[q] = 1
                self.outcomes[q] = np.random.binomial(
                    1, self.true_probabilities[q])
                for i in range(self.n):
                    self.outcome_payments[i, q] = self._get_score_winkler(i, q)

    def _get_vcg_allocation(self, ignore_i: int = -1) -> npt.NDArray:
        '''util for _elicit_vcg()'''
        assert isinstance(
            ignore_i, int), '_get_vcg_allocation(): ignoreRecommender must be int'

        solver = pulp.LpProblem('VCG_Allocation', pulp.LpMaximize)

        # ignore_i removes effect of recommender i
        weights = self.weights if ignore_i < 0 else np.delete(
            self.weights, ignore_i)
        reports = self.reports if ignore_i < 0 else np.delete(
            self.reports, ignore_i, 0)

        coeffs = np.append(weights.dot(reports),
                           # add liquidity no. of reserve borrowers have decision of threshold
                           np.full(self.liquidity, self.threshold))

        variables = [pulp.LpVariable(str(i), lowBound=0, upBound=1, cat='Integer')
                     for i in range(coeffs.size)]

        # objective
        solver += pulp.lpSum([variables[i] * coeffs[i]
                              for i in range(coeffs.size)])

        # constraints
        solver += pulp.lpSum(variables) <= self.liquidity

        solver.solve(pulp.PULP_CBC_CMD(msg=False))
        # print("Status: ", pulp.LpStatus[solver.status])

        allocation = np.array([variable.varValue for variable in variables],
                              dtype=np.int32)
        # remove reserve borrowers
        return allocation[:coeffs.size - self.liquidity]

    def _get_scores_vcg(self, allocation) -> npt.NDArray:
        return self.reports.dot(allocation) * self.weights

    def _elicit_vcg(self) -> None:
        self.allocation = self._get_vcg_allocation()
        scores = self._get_scores_vcg(self.allocation)

        # immediate payments
        for i in range(self.n):
            alt_allocation = self._get_vcg_allocation(ignore_i=i)
            alt_scores = self._get_scores_vcg(alt_allocation)
            self.immediate_payments[i] = np.sum(
                np.delete(alt_scores, i)) - np.sum(np.delete(scores, i))

        # outcome payments
        for q in range(self.m):
            assert self.allocation[q] == 0 or self.allocation[q] == 1, 'elicit_vcg(): Allocation must be binary'
            if self.allocation[q]:
                self.outcomes[q] = np.random.binomial(
                    1, self.true_probabilities[q])
                if self.outcomes[q]:
                    for i in range(self.n):
                        self.outcome_payments[i, q] = self.weights[i]

    def __str__(self):
        return '\n================================================\n\n' + \
            str(nullish(self.ELICITATION_STRATEGY, 'WARNING: DID NOT ELICIT')) + '\n' + \
            str(nullish(self.REPORT_STRATEGY, 'WARNING: NO REPORT STRATEGY')) + '\n\n' + \
            f'(n, m): {self.n, self.m}\n\n' + \
            f'borrower true probabilities:\n{self.true_probabilities}\n\n' + \
            f'recommender true beliefs:\n{self.true_beliefs}\n\n' + \
            f'recommender reports:\n{self.reports}\n\n' + \
            f'recommender weights:\n{self.weights}\n\n' + \
            f'lender threshold: {self.threshold}\n\n' + \
            f'lender liquidity: {self.liquidity}\n\n' + \
            f'lender allocation:\n{self.allocation}\n\n' + \
            f'borrower outcomes:\n{self.outcomes}\n\n' + \
            f'immediate payments:\n{self.immediate_payments}\n\n' + \
            f'outcome payments:\n{self.outcome_payments}'


def demo():
    # Default is uniformly random repayment probabilities, true beliefs that match
    # real probabilities, threshold of 0.5, equal recommender weights, unconstrained liquidity
    model = Model(n=5, m=3)
    model.add_beliefs_noise(BeliefNoise.GAUSSIAN)
    model.make_reports(ReportStrategy.TRUE)
    model.elicit(ElicitationStrategy.WINKLER)
    print(model)

    # You can also constrain liquidity, use VCG, and add a bunch of custom values
    model = Model(n=3,
                  m=4,
                  true_probabilities=[0.1, 0.4, 0.5, 0.7],
                  true_beliefs=[[0.1, 0.4, 0.5, 0.7],
                                [0.2, 0.5, 0.6, 0.8],
                                [0.05, 0.45, 0.3, 0.5]],
                  threshold=0.3,
                  weights=[0.7, 0.2, 0.1],
                  liquidity=1)
    model.add_beliefs_noise(BeliefNoise.GAUSSIAN)
    model.make_reports(ReportStrategy.TRUE)
    model.elicit(ElicitationStrategy.VCG)
    print(model)


def main():
    demo()


# UTILS

def truncate(x: np.ndarray, min: float = 0, max: float = 1) -> np.ndarray:
    '''performs double (min-max) truncation on all values in array x'''
    return np.minimum(max, np.maximum(min, x))


def nullish(x, y):
    '''nullish coalescing operator: return x if x is not null else y'''
    return x if x is not None else y


if __name__ == '__main__':
    main()
