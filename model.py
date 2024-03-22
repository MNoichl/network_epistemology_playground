from agent import Agent

class Model():
    def __init__(self, network, uncertainty, n_experiments):
        self.network = network
        self.uncertainty = uncertainty
        self.n_agents = len(network.nodes)
        self.n_experiments = n_experiments
        self.agents = [Agent(i) for i in range(self.n_agents)]
    
        
    def step(self):
        for agent in self.agents:
            agent.experiment(self.n_experiments, self.uncertainty)
        
        for agent in self.agents:
            neighbor_nodes = list(self.network.neighbors(agent.id))
            neighbor_agents = [self.agents[x] for x in neighbor_nodes]

            total_success= agent.n_success
            total_experiments = agent.n_experiments
            for neighbor in neighbor_agents:
                total_success += neighbor.n_success
                total_experiments += neighbor.n_experiments
            agent.bayes_update(total_success, total_experiments, self.uncertainty)