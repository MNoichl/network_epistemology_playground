
import dill
from network_utils import network_statistics, geometric_soft_configuration_graph_connected


def generate_parameters_randomization(_):

    p_rewiring = np.random.rand()
    # Epistemic Model parameters
    uncertainty = random.uniform(.005, .0001)
    n_experiments = random.randint(1, 50)
       
    
    # Do empirical sample
    n_agents = 943
    beta=1.44
    gamma=2.29
    mean_degree=7.357
    network  = geometric_soft_configuration_graph_connected(
                                                              beta=beta,
                                                              n=n_agents,
                                                              gamma=gamma,
                                                              mean_degree=mean_degree,
                                                              connect=True,
                                                              return_added_edges=False
                                                            )
    # Do randomization
    randomized_network = randomize_network(network, p_rewiring=p_rewiring)

    params_original = {
        "n_agents": n_agents,
        "network": network,
        "uncertainty": uncertainty,
        "n_experiments": n_experiments,
        "beta": beta,
        "gamma": gamma,
        "mean_degree": mean_degree,
        "p_rewiring": 0
    }
    stats = network_statistics(network)
    for stat in stats.keys():
      params_original[stat] = stats[stat]

    params_randomized = {
        "n_agents": n_agents,
        "network": randomized_network,
        "uncertainty": uncertainty,
        "n_experiments": n_experiments,
        "p_rewiring": p_rewiring
    }
    stats = network_statistics(randomized_network)
    for stat in stats.keys():
      params_randomized[stat] = stats[stat]

    # Merge both dictionaries
    params = {**params_original, **params_randomized}

    return params



# Use dill to deserialize the graph from a file
with open('perceptron_graph_pre_1979.pkl', 'rb') as f:
    G_perceptron = dill.load(f)

def generate_parameters_randomization_v2(_):

    p_rewiring = np.random.rand()
    # Epistemic Model parameters
    uncertainty = random.uniform(.005, .0001)
    n_experiments = random.randint(1, 50)
    
    # Randomize the perceptron
    # Do randomization
    real_randomized_network = randomize_network(G_perceptron, p_rewiring=p_rewiring)
    
    # Do empirical sample
    n_agents = 943
    beta=1.44
    gamma=2.29
    mean_degree=7.357
    sampled_network  = geometric_soft_configuration_graph_connected(
                                                              beta=beta,
                                                              n=n_agents,
                                                              gamma=gamma,
                                                              mean_degree=mean_degree,
                                                              connect=True,
                                                              return_added_edges=False
                                                            )
    # Do randomization
    sampled_randomized_network = randomize_network(sampled_network, p_rewiring=p_rewiring)

    # Now we want 4 networks: perceptron, perceptron randomized, sampled, sampled randomized

    params_real = {
        "n_agents": n_agents,
        "network": G_perceptron,
        "uncertainty": uncertainty,
        "n_experiments": n_experiments,
        "beta": beta,
        "gamma": gamma,
        "mean_degree": mean_degree,
        "p_rewiring": 0
    }
    stats = network_statistics(network)
    for stat in stats.keys():
      params_real[stat] = stats[stat]

    params_real_randomized = {
        "n_agents": n_agents,
        "network": real_randomized_network,
        "uncertainty": uncertainty,
        "n_experiments": n_experiments,
        "p_rewiring": p_rewiring
    }
    stats = network_statistics(real_randomized_network)
    for stat in stats.keys():
      params_real_randomized[stat] = stats[stat]

    params_sampled = {
        "n_agents": n_agents,
        "network": sampled_network,
        "uncertainty": uncertainty,
        "n_experiments": n_experiments,
        "beta": beta,
        "gamma": gamma,
        "mean_degree": mean_degree,
        "p_rewiring": 0
    }
    stats = network_statistics(sampled_network)
    for stat in stats.keys():
      params_sampled[stat] = stats[stat]

    params_sampled_randomized = {
        "n_agents": n_agents,
        "network": sampled_randomized_network,
        "uncertainty": uncertainty,
        "n_experiments": n_experiments,
        "p_rewiring": p_rewiring
    }
    stats = network_statistics(sampled_randomized_network)
    for stat in stats.keys():
      params_randomized[stat] = stats[stat]

    # Merge both dictionaries
    params = {**params_real, **params_real_randomized,**params_sampled,**params_sampled_randomized}

    return params