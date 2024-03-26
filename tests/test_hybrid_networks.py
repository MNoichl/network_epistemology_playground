import numpy as np

from hybrid_networks import create_hybrid_network


def test_create_hybrid_networks():
    net = create_hybrid_network(n_nodes=20, degree=6, p_preferential_attachment=0.7)
    assert len(list(net.nodes)) == 20
    out_degrees = np.array([net.out_degree[node] for node in net.nodes])
    assert all(out_degrees == 6)
