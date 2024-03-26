import numpy.random as rd


class Agent:
    """
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py
    Represents an agent in a network epistemology playground.

    Attributes:
    - credence (float): The agent's initial credence.
    - n_success (int): The number of successful experiments.
    - n_experiments (int): The total number of experiments.

    Methods:
    - __init__(self): Initializes the Agent object.
    - __str__(self): Returns a string representation of the Agent object.
    - experiment(self, n_experiments, uncertainty): Performs an experiment.
    - bayes_update(self, n_success, n_experiments, uncertainty): Updates the agent's
    credence using Bayes' rule.
    - jeffrey_update(self, neighbor, uncertainty, m): Updates the agent's credence
    using Jeffrey's rule.
    """

    def __init__(self, id):
        self.id = id
        self.credence: float = rd.uniform(0, 1)
        self.n_success: int = 0
        self.n_experiments: int = 0

    def __str__(self):
        return (
            f"credence = {round(self.credence, 2)}, n_success = {self.n_success}, "
            f"n_experiments = {self.n_experiments}"
        )

    def experiment(self, n_experiments: int, uncertainty: float):
        """
        Performs an experiment with the given parameters.

        Args:
        - n_experiments (int): The total number of experiments.
        - uncertainty (float): The uncertainty in the experiment.
        """
        if self.credence > 0.5:
            self.n_success = rd.binomial(n_experiments, 0.5 + uncertainty)
            self.n_experiments = n_experiments
        else:
            self.n_success = 0
            self.n_experiments = 0

    def bayes_update(self, n_success, n_experiments, uncertainty):
        """
        Updates the agent's credence using Bayes' rule.

        Args:
        - n_success (int): The number of successful experiments.
        - n_experiments: The total number of experiments.
        - uncertainty (float): The uncertainty in the experiment.
        """
        p_new_better = 0.5 + uncertainty
        p_new_worse = 0.5 - uncertainty
        n_failures = n_experiments - n_success
        credence_new_worse = 1 - self.credence
        likelihood_ratio_credence = credence_new_worse / self.credence
        likelihood_ratio_evidence_given_probability = (p_new_worse / p_new_better) ** (
            n_success - n_failures
        )
        self.credence = 1 / (
            1 + likelihood_ratio_credence * likelihood_ratio_evidence_given_probability
        )

    def jeffrey_update(self, neighbor, uncertainty, strength_update):
        """
        Updates the agent's credence using Jeffrey's rule.

        Args:
        - neighbor (Agent): An Agent object representing the neighbor agent.
        - uncertainty (float): The uncertainty in the experiment.
        - strength_update (float): The strength of the update.
        """
        # Todo (Hein): understand the update and refactor with sensible variable names
        n = neighbor.n
        k = neighbor.k

        p_E_H = (0.5 + uncertainty) ** k * (0.5 - uncertainty) ** (
            n - k
        )  # P(E|H)  = p^k (1-p)^(n-k)
        p_E_nH = (0.5 - uncertainty) ** k * (0.5 + uncertainty) ** (
            n - k
        )  # P(E|~H) = (1-p)^k p^(n-k)
        p_E = (
            self.credence * p_E_H + (1 - self.credence) * p_E_nH
        )  # P(E) = P(E|H) P(E) + P(E|~H) P(~H)

        p_H_E = self.credence * p_E_H / p_E  # P(H|E)  = P(H) P(E|H)  / P(E)
        p_H_nE = (
            self.credence * (1 - p_E_H) / (1 - p_E)
        )  # P(H|~E) = P(H) P(~E|H) / P(~E)

        # q_E = max(1 - abs(self.credence - neighbor.credence) * m * (1 - p_E), 0)
        # O&W's Eq. 1 (anti-updating)
        q_E = 1 - min(1, abs(self.credence - neighbor.credence) * strength_update) * (
            1 - p_E
        )  # O&W's Eq. 2

        self.credence = p_H_E * q_E + p_H_nE * (
            1 - q_E
        )  # Jeffrey's Rule # P'(H) = P(H|E) P'(E) + P(H|~E) P'(~E)#


class BetaAgent(Agent):
    """Inspired by Zollman, Kevin J. S. 2010. The Epistemic Benefit of Transient
    Diversity. Erkenntnis 72 (1): 17--35. https://doi.org/10.1007/s10670-009-9194-6.
    (Especially sections 2 and 3.)

    Todo (Hein): I'm not sure whether we should use one pair (alpha,beta) or
    two pairs (alpha_i,beta_i)."""

    def __init__(self, id):
        Agent.__init__(self, id)
        self.alpha: int = rd.randint(0, 4) #Why are these integers? (MN)
        self.beta: int = rd.randint(0, 4)
        if (self.alpha, self.beta) == (0, 0):
            self.credence = 0.5
        else:
            self.credence: float = self.alpha / (self.alpha + self.beta)

    def __str__(self):
        return (
            f"credence = {round(self.credence, 2)}, n_success = {self.n_success},"
            f"n_experiments = {self.n_experiments}, alpha = {self.alpha},"
            f"beta = {self.beta}"
        )

    def beta_update(self, n_success, n_experiments, uncertainty):
        """
        Updates the agent's beta distribution and credence.

        Args:
        - n_success (int): The number of successful experiments.
        - n_experiments: The total number of experiments.
        - uncertainty (float): The uncertainty in the experiment.
        """
        # Todo (Hein): Note that this update does not depend on uncertainty
        # whereas the bayes_update above does. Did we make a mistake?
        n_failures = n_experiments - n_success
        self.alpha += n_success
        self.beta += n_failures
        if n_experiments > 0:
            self.credence = self.alpha / (self.alpha + self.beta)
