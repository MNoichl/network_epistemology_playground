import numpy as np
import numpy.random as rd


class UncertaintyProblem:
    """
    The problem of theory choice involves two theories where the new_theory is better
    by the margin of uncertainty.

    Attributes:
    - uncertainty (float): The uncertainty in the theory choice.

    Methods
    - experiment(self, n_experiments): Performs an experiment using the new_theory.
    """

    def __init__(self, uncertainty: float = 0.001) -> None:
        self.uncertainty = uncertainty
        self.p_old_theory = 0.5
        self.p_new_theory = 0.5 + uncertainty
        
    def experiment(self, n_experiments: int):
        """
        Performs an experiment using the new_theory.

        Args:
        - n_experiments (int): the number of experiments.
        """
        n_success = rd.binomial(n_experiments, self.p_new_theory)
        self.successes += n_success
        self.failures += (n_experiments-n_success)
        return n_success, n_experiments


class Agent:
    """
    An agent in a network epistemology playground, either Bayesian or Jeffreyan.
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py

    Attributes:
    - credence (float): The agent's initial credence that the new theory is better.
    - n_success (int): The number of successful experiments.
    - n_experiments (int): The total number of experiments.

    Methods:
    - __init__(self): Initializes the Agent object.
    - __str__(self): Returns a string representation of the Agent object.
    - experiment(self, n_experiments, uncertainty): Performs an experiment.
    - bayes_update(self, n_success, n_experiments, uncertainty): Updates the agent's
    credence using Bayes' rule.
    - jeffrey_update(self, neighbor, uncertainty, mistrust_rate): Updates the agent's
    credence using Jeffrey's rule.
    """

    def __init__(self, id, uncertainty_problem: UncertaintyProblem):
        self.id = id
        self.uncertainty_problem = uncertainty_problem
        self.credence: float = rd.uniform(0, 1)
        # I initialize with 1 rather than zero so that we can sample from the beta
        self.n_success: int = 1
        self.n_experiments: int = 1
        self.accumulated_successes = np.zeros(1)+1
        self.accumulated_failures = np.zeros(1)+1

    def __str__(self):
        return (
            f"credence = {round(self.credence, 2)}, n_success = {self.n_success}, "
            f"n_experiments = {self.n_experiments}"
        )

    def experiment(self, n_experiments: int):
        """
        Performs an experiment with the given parameters.

        Args:
        - n_experiments (int): The total number of experiments.
        - uncertainty (float): The uncertainty in the experiment.
        """
        if self.credence > 0.5:
            self.n_success, self.n_experiments = self.uncertainty_problem.experiment(
                n_experiments
            )
            self.accumulated_successes += self.n_success
            self.accumulated_failures += (self.n_experiments-self.n_success)
        else:
            self.n_success = 0
            self.n_experiments = 0

    def bayes_update(self, n_success, n_experiments):
        """
        Updates the agent's credence using Bayes' rule. The basic setting is that the
        agent knows the probability of an old theory but does not know the probability
        of a new theory. The probability of the new theory is assumed to be either
        0.5 + uncertainty or 0.5 - uncertainty.

        Args:
        - n_success (int): The number of successful experiments.
        - n_experiments: The total number of experiments.
        """
        p_new_better = 0.5 + self.uncertainty_problem.uncertainty
        p_new_worse = 0.5 - self.uncertainty_problem.uncertainty
        n_failures = n_experiments - n_success
        credence_new_worse = 1 - self.credence
        likelihood_ratio_credence = credence_new_worse / self.credence
        likelihood_ratio_evidence_given_probability = (p_new_worse / p_new_better) ** (
            n_success - n_failures
        )
        self.credence = 1 / (
            1 + likelihood_ratio_credence * likelihood_ratio_evidence_given_probability
        )

    def beta_update(self,n_success,n_experiments):
        p_new_better = 0.5 + self.uncertainty_problem.uncertainty
        p_new_worse = 0.5 - self.uncertainty_problem.uncertainty   
        mean, var= beta.stats(self.accumulated_successes, self.accumulated_failures, moments='mv')
        self.credence = mean

    def jeffrey_update(self, neighbor, uncertainty, mistrust_rate):
        """
        Updates the agent's credence using Jeffrey's rule.

        Args:
        - neighbor (Agent): An Agent object representing the neighbor agent.
        - uncertainty (float): The uncertainty in the experiment.
        - mistrust_rate (float): The rate at which difference of opinion increases
        discounting.
        """
        # Todo (Hein): understand the update and refactor with sensible variable names
        n_experiments = neighbor.n_experiment
        n_success = neighbor.n_success
        n_failures = n_experiments - n_success

        p_success_given_new_better = 0.5 + uncertainty
        p_E_given_new_better = (
            p_success_given_new_better**n_success
            * (1 - p_success_given_new_better) ** n_failures
        )  # P(E|H)  = p^k (1-p)^(n-k)
        p_success_given_new_worse = 0.5 - uncertainty
        p_E_given_new_worse = (
            p_success_given_new_worse**n_success
            * (1 - p_success_given_new_worse) ** n_failures
        )  # P(E|~H) = (1-p)^k p^(n-k)
        p_E = (
            self.credence * p_E_given_new_better
            + (1 - self.credence) * p_E_given_new_worse
        )  # P(E) = P(E|H) P(H) + P(E|~H) P(~H)

        p_new_better_given_E = (
            self.credence * p_E_given_new_better / p_E
        )  # P(H|E)  = P(H) P(E|H)  / P(E)
        p_new_worse_given_E = (
            self.credence * (1 - p_E_given_new_better) / (1 - p_E)
        )  # P(H|~E) = P(H) P(~E|H) / P(~E)

        # p_post_E = max(1 - abs(self.credence - neighbor.credence) * mistrust_rate *
        # (1 - p_E), 0)
        # O&W's Eq. 1 (anti-updating)
        p_post_E = 1 - min(
            1, abs(self.credence - neighbor.credence) * mistrust_rate
        ) * (
            1 - p_E
        )  # O&W's Eq. 2

        self.credence = p_new_better_given_E * p_post_E + p_new_worse_given_E * (
            1 - p_post_E
        )  # Jeffrey's Rule # P'(H) = P(H|E) P'(E) + P(H|~E) P'(~E)#


class Bandit:
    def __init__(self, p_theories=None):
        if p_theories is None:
            self.n_theories = 2
            self.p_theories = np.random.random(2)
        if p_theories is not None:
            self.n_theories = len(p_theories)
            self.p_theories = p_theories

    def experiment(self, theory, n_experiments):
        p_theory = self.p_theories[theory]
        n_success = rd.binomial(n_experiments, p_theory)
        return n_success, n_experiments


class BetaAgent:
    """Inspired by Zollman, Kevin J. S. 2010. The Epistemic Benefit of Transient
    Diversity. Erkenntnis 72 (1): 17--35. https://doi.org/10.1007/s10670-009-9194-6.
    (Especially sections 2 and 3.)

    Attributes:
    - id: The id of the BetaAgent
    - beliefs (np.array): The beliefs of the agent. Each index of the array represents a
    theory and contains an array the form [alpha (float), beta (float)]
    representing the beta-distribution that models the agent's beliefs about that
    theory.
    - experiment_result (np.array): The result of the agent's last experiment.
    Each index of the array represents a theory and contains an array of the form
    [n_success (int), n_experiments (int)] representing the result of the experiment on
    that theory, if any. If there is no experiment on a given theory, then the result of
    the experiment on that theory is [0, 0].

    Methods:
    - __init__(self): Initializes the BetaAgent object.
    - __str__(self): Returns a string representation of the BetaAgent object.
    - n_theories (int): The number of theories under consideration.
    - experiment(self, n_experiments, p_theories): Performs an experiment and updates
    the agent's experiment_result.
    - beta_update(self, experiment_results): Updates the agent's beliefs on the basis of
    experiments. Experiments are represented by an array, where each index of the array
    represents a theory and contains an array of the form [n_success (int),
    n_experiments (int)] representing the result of the experiments.
    """

    def __init__(self, id, bandit: Bandit):
        self.id = id
        self.bandit = bandit
        self.n_theories = bandit.n_theories
        self.beliefs: np.array = np.array(
            [[rd.random(), rd.random()] for _ in range(self.n_theories)]
        )
        self.experiment_result: np.array = np.array(
            [[0, 0] for _ in range(self.n_theories)]
        )

    def __str__(self):
        return (
            # f"credence = {round(self.credence, 2)}, n_success = {self.n_success},"
            # f"n_experiments = {self.n_experiments}, alpha = {self.alpha},"
            # f"beta = {self.beta}"
        )

    def experiment(self, n_experiments: int):
        """Performs an experiment and updates the agent's experiment_result.

        Args:
        - n_experiments (int): The number of experiments.
        - p_theories (np.array): The probabilities of success, one for each theory."""
        # Reset experiment_result
        self.experiment_result = np.array([[0, 0] for _ in range(self.n_theories)])

        decision = self.decision()

        # Perform experiment on that theory and update experiment_result
        n_success, n_experiments = self.bandit.experiment(decision, n_experiments)
        self.experiment_result[decision] = [n_success, n_experiments]

    def decision(self):
        credences = np.array(
            [
                self.beliefs[theory_id][0]
                / (self.beliefs[theory_id][0] + self.beliefs[theory_id][1])
                for theory_id in range(self.n_theories)
            ]
        )
        return rd.choice(np.flatnonzero(credences == np.max(credences)))

    def beta_update(self, experiment_results):
        """Updates the agent's beliefs based on experiment_results.

        Args:
        - experiment_results (np.array): An array representing the results from
        experiments represented in an array.
        """
        for theory in range(self.n_theories):
            n_success = experiment_results[theory][0]
            n_experiments = experiment_results[theory][1]
            n_failures = n_experiments - n_success
            self.beliefs[theory][0] += n_success
            self.beliefs[theory][1] += n_failures
