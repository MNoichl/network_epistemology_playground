import numpy as np
import numpy.random as rd
from scipy.stats import beta

class UncertaintyProblem:
    """
    The problem of theory choice involves two theories where the new_theory is better
    by the margin of uncertainty.

    Attributes:
    - uncertainty (float): The uncertainty in the theory choice.

    Methods
    - experiment(self, n_experiments): Performs an experiment using the new_theory.
    """

    def __init__(self, uncertainty: float = 0.1) -> None:
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
        # I initialize with 1 rather than zero so that we can sample from the beta
        self.n_success: int = 1
        self.n_experiments: int = 1
        # For the beta agent
        self.accumulated_successes = np.zeros(1)+1
        self.accumulated_failures = np.zeros(1)+1      
        self.choice_history = []
        self.credence: float = rd.uniform(0, 1)
        self.credence_history = []
        self.credence_history.append(self.credence)
        self.inner_perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
        self.epsilon=0.1
        
    def init_bayes(self):
        self.credence: float = rd.uniform(0, 1)
        self.credence_history = []
        self.credence_history.append(self.credence)
    # Instead of initializing with just alpha=beta=1, I ALSO initialize be sampling from the binomial/uncertainty problem
    def init_beta(self):
        n_success, n_experiments = self.uncertainty_problem.experiment(2)
        self.accumulated_successes+=n_success
        self.accumulated_failures+=(n_experiments-n_success)
        mean, var= beta.stats(self.accumulated_successes, self.accumulated_failures, moments='mv')
        #print(self.accumulated_successes/(self.accumulated_failures+self.accumulated_successes))
        self.credence = mean[0]
        self.credence_history = []
        self.credence_history.append(self.credence)

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
            self.choice_history.append(1)
        else:
            self.n_success = 0
            self.n_experiments = 0
            self.choice_history.append(0)

    # I am not sure adding epsilon greedy helps because in this case all models will
    # achieve correct true consensus
    def egreedy_experiment(self, n_experiments: int):
        if np.random.rand() < self.epsilon:
            self.n_success, self.n_experiments = self.uncertainty_problem.experiment(
                n_experiments
            )
            self.choice_history.append(1)
        else:
            self.experiment(n_experiments)
            
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
        self.credence_history.append(self.credence)

    def beta_update(self,n_success,n_experiments):
        self.accumulated_successes += n_success
        self.accumulated_failures += (n_experiments-n_success)
        p_new_better = 0.5 + self.uncertainty_problem.uncertainty
        p_new_worse = 0.5 - self.uncertainty_problem.uncertainty   
        mean, var= beta.stats(self.accumulated_successes, self.accumulated_failures, moments='mv')
        self.credence = mean[0]
        self.credence_history.append(self.credence) # this is usually a vector to factor multiple theories

    def perceptron_update(self,n_success,n_experiments):
        # Update inner variables
        self.accumulated_successes += n_success
        self.accumulated_failures += (n_experiments-n_success)
        n_failures = n_experiments - n_success
        # train the inner perceptron with incoming data
        success_rate = n_success/n_experiments #this will be our label
        training_input = np.array([[n_success,n_failures]])
        label = np.array([success_rate])
        self.inner_perceptron.train(training_input, label)
        # Now we need an estimate for the credence
        # let it be the prediction over the accumulated information
        #print(type(self.accumulated_successes))
        test_input = np.array([self.accumulated_successes[0], self.accumulated_failures[0]])
        #print(test_input.shape)
        big_prediction = self.inner_perceptron.predict(test_input)
        self.credence = big_prediction
        self.credence_history.append(self.credence)
        
    def jeffrey_update(self, neighbor, uncertainty, mistrust_rate=0.5):
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
        self.credence_history.append(self.credence)
    
    # I am playing with Heins code here, the original is above (Ignacio)
    # Part of the problem is that the jeffrey update collides with what the model is doing
    # in the model what we have is that each agent receives the cummulative number of successes and failures from their neighbors as input
    # this version of jeffrey update seems to require as input a neighbor. It is doable though, but I will let it sleep for a bit.
    def jeffrey_updatev2(self, neighbor_n_success, neigbor_n_failures, neighbor_credence,mistrust_rate=0.5):
        """
        Updates the agent's credence using Jeffrey's rule.

        Args:
        - neighbor (Agent): An Agent object representing the neighbor agent.
        - uncertainty (float): The uncertainty in the experiment.
        - mistrust_rate (float): The rate at which difference of opinion increases
        discounting.
        """
        #print(type(neighbor_n_success))
        # Todo (Hein): understand the update and refactor with sensible variable names
        p_success_given_new_better = 0.5 + self.uncertainty_problem.uncertainty
        #print(type(p_success_given_new_better))
        p_E_given_new_better = (
            p_success_given_new_better**neighbor_n_success
            * (1 - p_success_given_new_better) ** neigbor_n_failures
        )  # P(E|H)  = p^k (1-p)^(n-k)
        p_success_given_new_worse = 0.5 - self.uncertainty_problem.uncertainty
        p_E_given_new_worse = (
            p_success_given_new_worse**neighbor_n_success
            * (1 - p_success_given_new_worse) ** neigbor_n_failures
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
            1, abs(self.credence - neighbor_credence) * mistrust_rate
        ) * (
            1 - p_E
        )  # O&W's Eq. 2

        self.credence = p_new_better_given_E * p_post_E + p_new_worse_given_E * (
            1 - p_post_E
        )  # Jeffrey's Rule # P'(H) = P(H|E) P'(E) + P(H|~E) P'(~E)#
        self.credence_history.append(self.credence)
        
        
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



# I have a payground here where we use a 'perceptron' agent, which estimates the mean given successes and failures

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid function

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

# Example usage:
if False:#__name__ == "__main__":
    # Initialize the perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=1000)

    # Training data (2 integers as inputs, float as output)
    training_inputs = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])
    labels = np.array([0.5, 0.6, 0.7, 0.8])  # Example float outputs

    # Train the perceptron
    perceptron.train(training_inputs, labels)

    # Test the perceptron
    test_input = np.array([2, 3])
    print("Prediction for input [2, 3]:", perceptron.predict(test_input))
