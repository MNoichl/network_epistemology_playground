import networkx as nx
import numpy as np

from model import Model


def test_init():
    network = nx.erdos_renyi_graph(10, 0.5)
    model1 = Model(network, 0.01, 10, "beta")
    credences = np.array([agent.credence for agent in model1.agents])
    assert np.all(credences <= 1)
    assert np.all(credences >= 0)
    successes_array = np.array([agent.n_success for agent in model1.agents])
    assert np.all(successes_array == 0)
    experiments_array = np.array([agent.n_experiments for agent in model1.agents])
    assert np.all(experiments_array == 0)


def test_agents_experiment():
    network = nx.erdos_renyi_graph(10, 0.5)
    model1 = Model(network, 0.01, 10, "beta")
    model1.agents_experiment()
    experiments_array = np.array(
        [agent.n_experiments for agent in model1.agents if agent.credence > 0.5]
    )
    assert np.all(experiments_array != 0)


def test_agents_update():
    pass


def test_step():
    network = nx.erdos_renyi_graph(10, 0.5)
    model1 = Model(network, 0.01, 10, "beta")
    model1.step()


def test_simulation():
    network = nx.erdos_renyi_graph(10, 0.5)
    model1 = Model(network, 0.01, 10, "beta")
    model1.simulation()
