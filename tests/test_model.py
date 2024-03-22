from model import Model
from agent import Agent
import networkx as nx

def test_init():
    network = nx.erdos_renyi_graph(10, 0.5)
    model1 = Model(network, 0.01, 10)
    agent1 = model1.agents[0]
    assert agent1.credence <= 1
    assert agent1.credence >= 0
    assert agent1.n_success == 0
    assert agent1.n_experiments == 0