import numpy as np
import tqdm

from agent import Agent, BetaAgent


class Model:
    """
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py
    Represents an agent in a network epistemology playground.

    Attributes:
    - network: The network.
    - n_experiments (int): The number of experiments per step.
    - uncertainty (float): The uncertainty in the experiment.
    - agent_type (str): The type of agents, "bayes", "beta" or "jeffrey"

    Methods:
    - __init__(self): Initializes the Model object.
    - __str__(self): Returns a string representation of the Model object.
    - simulation(self, number_of_steps): Runs a simulation of the model.
    - step(self): Updates the model with one step, consisting of experiments and
    updates.
    - agents_experiment(self): Updates the model with one round of experiments.
    - agents_update(self): Updates the model with one round of updates.
    """

    def __init__(
        self,
        network,
        uncertainty: float,
        n_experiments: int,
        agent_type: str,
        *args,
        **kwargs
    ):
        self.network = network
        self.uncertainty = uncertainty
        self.n_agents = len(network.nodes)
        self.n_experiments = n_experiments
        self.agent_type = agent_type
        if self.agent_type == "beta":
            self.agents = [BetaAgent(i) for i in range(self.n_agents)]
        else:
            self.agents = [Agent(i) for i in range(self.n_agents)]

    def run_simulation(
        self, number_of_steps: int = 10**6, show_bar: bool = False, *args, **kwargs
    ):
        """Runs a simulation of the model and sets model.conclusion.

        Args:
            number_of_steps (int, optional): Number of steps in the simulation
            (it will end sooner if the stop condition is met). Defaults to 10**6."""

        def stop_condition(credences_prior, credences_post) -> bool:
            # if np.all((credences_post < 0.5) or (credences_post > 0.99)):
            if np.all(credences_post < 0.5) or np.all(credences_post > 0.99):
                return True
            # if np.allclose(credences_prior, credences_post):
            #     return True
            return False

        iterable = range(number_of_steps)

        if show_bar:
            iterable = tqdm.tqdm_notebook(iterable)

        alternative_stop = False
        self.conclusion_alternative_stop = False
        for _ in iterable:
            credences_prior = np.array([agent.credence for agent in self.agents])
            self.step()
            credences_post = np.array([agent.credence for agent in self.agents])
            if not alternative_stop:
                if np.allclose(credences_prior, credences_post):
                    alternative_stop = True
                    self.conclusion_alternative_stop = np.all(credences_post > 0.99)
            if stop_condition(credences_prior, credences_post):
                break
        self.conclusion = np.all(credences_post > 0.99)
        if not alternative_stop:
            self.conclusion_alternative_stop = self.conclusion

    def step(self):
        """Updates the model with one step, consisting of experiments and updates"""
        self.agents_experiment()
        self.agents_update()

    def agents_experiment(self):
        for agent in self.agents:
            agent.experiment(self.n_experiments, self.uncertainty)

    def agents_update(self):
        for agent in self.agents:
            # gather information from neighbors
            neighbor_nodes = list(self.network.neighbors(agent.id))
            neighbor_agents = [self.agents[x] for x in neighbor_nodes]
            total_success = agent.n_success
            total_experiments = agent.n_experiments
            for neighbor in neighbor_agents:
                total_success += neighbor.n_success
                total_experiments += neighbor.n_experiments

            # update
            if self.agent_type == "beta":
                agent.beta_update(total_success, total_experiments, self.uncertainty)
            elif self.agent_type == "bayes":
                agent.bayes_update(total_success, total_experiments, self.uncertainty)
            elif self.agent_type == "jeffrey":
                agent.jeffrey_update(total_success, total_experiments, self.uncertainty)
