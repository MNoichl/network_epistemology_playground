import numpy as np
import tqdm

from agent import Agent, Bandit, BetaAgent, UncertaintyProblem


class Model:
    """
    Adapted from https://github.com/jweisber/sep-sen/blob/master/bg/agent.py
    Represents an agent in a network epistemology playground.

    Attributes:
    - network: The network.
    - n_experiments (int): The number of experiments per step.
    - agent_type (str): The type of agents, "bayes", "beta" or "jeffrey"
    - uncertainty (float): The uncertainty in the experiment.
    - p_theories (list): The success probabilities of the theories.

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
        n_experiments: int,
        agent_type: str,
        uncertainty: float = None,
        p_theories: list = None,
        *args,
        **kwargs
    ):
        self.network = network
        self.n_agents = len(network.nodes)
        self.n_experiments = n_experiments
        self.agent_type = agent_type
        #if self.agent_type == "beta":
            #self.bandit = Bandit(p_theories)
            # self.agents = [BetaAgent(i, self.bandit) for i in range(self.n_agents)]

        # else:
        self.uncertainty_problem = UncertaintyProblem(uncertainty)
        self.agents = [
            Agent(i, self.uncertainty_problem) for i in range(self.n_agents)
        ]

    def run_simulation(
        self, number_of_steps: int = 10**6, show_bar: bool = False, *args, **kwargs
    ):
        """Runs a simulation of the model and sets model.conclusion.

        Args:
            number_of_steps (int, optional): Number of steps in the simulation
            (it will end sooner if the stop condition is met). Defaults to 10**6."""

        # Weisberg's stopping condition:
        # def stop_condition(credences_prior, credences_post) -> bool:
        #     if np.all(credences_post < 0.5) or np.all(credences_post > 0.99):
        #         return True
        #     return False

        # Weisberg's true_consensus condition
        # def true_consensus_condition(credences: np.array) -> bool:
        #     return all(credences > 0.99)

        def stop_condition(credences_prior, credences_post) -> bool:
            return np.allclose(credences_prior, credences_post)

        def true_consensus_condition(credences: np.array) -> float:
            return (credences > 0.5).mean()

        iterable = range(number_of_steps)

        if show_bar:
            iterable = tqdm.tqdm_notebook(iterable)

        alternative_stop = False
        self.conclusion_alternative_stop = False
        for _ in iterable:
            credences_prior = np.array([agent.credence for agent in self.agents])
            self.step()
            credences_post = np.array([agent.credence for agent in self.agents])
            # if not alternative_stop:
            #     if alternative_stop_condition(credences_prior, credences_post):
            #         alternative_stop = True
            #         self.conclusion_alternative_stop = true_consensus_condition(
            #             credences_post
            #         )
            if stop_condition(credences_prior, credences_post):
                self.conclusion = true_consensus_condition(credences_post)
                if not alternative_stop:
                    self.conclusion_alternative_stop = self.conclusion
                break
            self.conclusion = true_consensus_condition(
            credences_post
        )  # We should set this even if we don't break, right??? - MN
        
        self.add_agent_history()
            

    def step(self):
        """Updates the model with one step, consisting of experiments and updates."""
        self.agents_experiment()
        self.agents_update()

    def agents_experiment(self):
        for agent in self.agents:
            agent.experiment(self.n_experiments)

    def agents_update(self):
        for agent in self.agents:
            # gather information from neighbors
            neighbor_nodes = list(self.network.neighbors(agent.id))
            neighbor_agents = [self.agents[x] for x in neighbor_nodes] #Double check for bipartite graphs?!
            total_success = agent.n_success
            total_experiments = agent.n_experiments
            for neighbor in neighbor_agents:
                total_success += neighbor.n_success
                total_experiments += neighbor.n_experiments

            # update
            if self.agent_type == "beta":
                agent.beta_update(total_success, total_experiments)
            elif self.agent_type == "bayes":    
                agent.bayes_update(total_success, total_experiments)
            elif self.agent_type == "jeffrey":
                for neighbor in neighbor_agents: # I am here copying what Weisberg did
                    if neighbor.id==agent.id:
                        agent.bayes_update(agent.n_success, agent.n_experiments)
                    else:
                        neighbor_n_success = int(neighbor.n_success)
                        neighbor_n_experiments=int(neighbor.n_experiments)
                        #print(type(neighbor_n_success))
                        #print(type(neighbor_n_experiments))
                        neigbor_n_failures = neighbor_n_experiments - neighbor_n_success
                        #print(type(neigbor_n_failures))
                        neighbor_credence = neighbor.credence
                        agent.jeffrey_updatev2(neighbor_n_success, neigbor_n_failures,neighbor_credence)
                
                
    def add_agent_history(self):
        self.agent_histories = [agent.credence_history for agent in self.agents]
