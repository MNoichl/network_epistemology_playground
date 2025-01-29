import networkx as nx
import numpy as np
import time

def geometric_soft_configuration_graph_connected(
    beta=1.5,
    n=100,
    gamma=2.7,
    mean_degree=5,
    connect=True,
    return_added_edges=False
):
    """
    Generates a geometric soft configuration model graph using NetworkX.
    Optionally ensures the graph is fully connected by adding minimal edges.
    Can also return the number of edges added to achieve connectivity.

    Parameters:
    - beta (float): The spatial distance scaling exponent.
    - n (int): Number of nodes in the graph.
    - gamma (float): The power-law exponent of the degree distribution.
    - mean_degree (float): The desired average degree of the graph.
    - connect (bool): If True, ensures the graph is connected by adding edges.
    - return_added_edges (bool): If True, returns the number of edges added.

    Returns:
    - G (networkx.Graph): The generated graph (connected or unconnected).
    - added_edges (int, optional): The number of edges added to make the graph connected.
      Only returned if return_added_edges is True.
    """
    # Generate the initial graph using the NetworkX function
    G = nx.geometric_soft_configuration_graph(
        n=n, gamma=gamma, beta=beta, mean_degree=mean_degree
    )

    added_edges = 0  # Counter for the number of edges added

    if connect:
        # Check if the graph is connected
        if not nx.is_connected(G):
            # Find all connected components
            components = list(nx.connected_components(G))

            # Iterate over components and connect them
            for i in range(len(components) - 1):
                # Select one node from the current component
                u = next(iter(components[i]))
                # Select one node from the next component
                v = next(iter(components[i + 1]))
                # Add an edge between the two nodes
                G.add_edge(u, v)
                added_edges += 1  # Increment the counter

                # Optionally, merge the two components for the next iteration
                components[i + 1] = components[i].union(components[i + 1])

    if return_added_edges:
        return G, added_edges
    else:
        return G

#G, num_added_edges = geometric_soft_configuration_graph_connected(
#    beta=1.5, n=100, gamma=2.7, mean_degree=5, connect=True, return_added_edges=True
#)

#print(f"Number of edges added to connect the graph: {num_added_edges}")
#print(f"Is the graph connected? {nx.is_connected(G)}")






def calculate_degree_gini(degrees):
    # Sort the degrees in ascending order
    sorted_degrees = sorted(degrees)
    n = len(degrees)

    # Calculate the cumulative sum of the sorted degrees
    cumulative_degrees = sum(sorted_degrees)

    # Calculate the Gini coefficient
    gini_numerator = 0
    for i, degree in enumerate(sorted_degrees):
        gini_numerator += (i + 1) * degree

    gini_denominator = n * cumulative_degrees

    # Gini formula
    gini_coefficient = (2 * gini_numerator) / gini_denominator - (n + 1) / n

    return gini_coefficient


def alternative_calculate_degree_gini(degrees):
    """Using numpy for speed."""
    # Sort the degrees using numpy
    sorted_degrees = np.sort(degrees)
    n = len(degrees)
    
    # Calculate cumulative sum using numpy
    cumsum = np.cumsum(sorted_degrees, dtype=float)
    
    # Simplified Gini formula using numpy operations
    gini_coefficient = (n + 1 - 2 * np.sum((n - np.arange(n)) * sorted_degrees) / np.sum(sorted_degrees)) / n
    
    return gini_coefficient




def network_statistics(G):
    stats = {}

    # Number of nodes and edges#
#    stats['number_of_nodes'] = G.number_of_nodes()
#    stats['number_of_edges'] = G.number_of_edges()

    # Average degree
    degrees = [deg for _, deg in G.degree()]
    stats['average_degree'] = sum(degrees) / len(degrees)

    # Gini coefficient
    #print(degrees)
    stats['degree_gini_coefficient'] = calculate_degree_gini(degrees)

    # Approximate average clustering coefficient
    stats['approx_average_clustering_coefficient'] = nx.average_clustering(G)#, trials=50000)

    # Calculate the diameter (approximate)
    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
    else:
        largest_component = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_component)
        stats['diameter'] = nx.diameter(subgraph)

    # Add additional metrics as needed here, e.g., centrality measures

    return stats








def directed_network_statistics(G):
    """
    Computes network statistics adapted for directed graphs, handling multigraphs 
    by converting them to simple graphs for certain calculations.

    This version reports:
    - Average degree (in/out are equal in total, so we only report one)
    - Gini coefficients for both in-degree and out-degree distributions
    - Average clustering coefficient of the underlying undirected version
    - Weak diameter (diameter of the largest weakly connected component, considered undirected)

    Parameters:
    ----------
    G : networkx.DiGraph or networkx.MultiDiGraph
        A directed graph or directed multigraph

    Returns:
    -------
    dict
        Dictionary containing:
        - 'average_degree': float
            Average degree (in or out, they're equal)
        - 'in_gini_coefficient': float
            Gini coefficient of the in-degree distribution
        - 'out_gini_coefficient': float
            Gini coefficient of the out-degree distribution
        - 'approx_average_clustering_coefficient': float
            Average clustering coefficient
        - 'weak_diameter': int
            Diameter of the largest weakly connected component
    """
    stats = {}

    # Get degrees
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    
    # Average degree (in/out are equal in directed graphs)
    stats['average_degree'] = sum(out_degrees) / len(out_degrees) if len(out_degrees) > 0 else 0

    # Gini coefficients for both in and out degree distributions
    stats['in_gini_coefficient'] = alternative_calculate_degree_gini(in_degrees)
    stats['out_gini_coefficient'] = alternative_calculate_degree_gini(out_degrees)

    # Convert to simple graph for clustering calculations if it's a multigraph
    if G.is_multigraph():
        simple_G = nx.DiGraph(G)  # Convert to simple directed graph
    else:
        simple_G = G
        
    # Get undirected version for clustering
    undirected_G = simple_G.to_undirected()
    stats['approx_average_clustering_coefficient'] = nx.average_clustering(undirected_G)

    # Weak diameter: diameter of the largest weakly connected component (undirected)
    if nx.is_empty(G):
        stats['weak_diameter'] = 0
    else:
        largest_weak_component = max(nx.weakly_connected_components(G), key=len)
        subgraph = G.subgraph(largest_weak_component).to_undirected()
        if subgraph.number_of_nodes() > 1:
            stats['weak_diameter'] = nx.diameter(subgraph)
        else:
            stats['weak_diameter'] = 0

    return stats