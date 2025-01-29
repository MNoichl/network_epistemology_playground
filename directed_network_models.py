import networkx as nx
import random
import numpy as np
import math
import scipy.stats as st
import numbers


# A collection of directed network models

def directed_watts_strogatz_graph(L, K, beta, self_loops=False, seed=None):
    """
    Creates a directed Watts–Strogatz (WS) network using
    the distance-dependent probability formulation from Song & Wang (2014).
    
    Parameters
    ----------
    L : int
        Number of nodes in the graph.
    K : int
        Equivalent to the 'mean degree' in the ring-lattice limit
        (must be <= L-1 if self_loops=False).
    beta : float in [0, 1]
        "Rewiring" parameter. 0 -> regular ring-lattice limit,
        1 -> directed Erdős–Rényi limit.
    self_loops : bool
        If True, allows edges i->i. If False, disallows them.
    seed : int or None
        Random seed for reproducibility (optional).
    
    Returns
    -------
    G : nx.DiGraph
        A directed graph with L nodes, generated according to the
        distance-dependent WS rule.
    """
    if seed is not None:
        random.seed(seed)
    
    # Create an empty DiGraph
    G = nx.DiGraph()
    G.add_nodes_from(range(L))
    
    # p0 is the base connection probability in the random (beta=1) limit
    # If you allow self-loops, then "L" is the denominator.
    # If you disallow self-loops, then "L - 1" is the denominator.
    denom = L if self_loops else (L - 1)
    p0 = K / denom
    
    # For distance calculations, define half_L = floor(L/2),
    # so ring distance D_ij in {0, 1, 2, ..., half_L}.
    # Then normalized distance d_ij = D_ij / half_L.
    half_L = L // 2  # floor(L/2)
    
    for i in range(L):
        for j in range(L):
            if (not self_loops) and (i == j):
                continue
            
            # Ring distance (shortest steps around the circle)
            D_ij = abs(i - j)
            if D_ij > half_L:
                D_ij = L - D_ij
            
            # Normalized distance
            d_ij = D_ij / half_L if half_L > 0 else 0.0
            
            # Probability p_ij = beta * p0 + (1 - beta) * indicator(d_ij <= p0)
            # The 'indicator' is 1 if p0 >= d_ij, else 0
            if d_ij <= p0:
                p_ij = beta * p0 + (1 - beta)
            else:
                p_ij = beta * p0
            
            # Bernoulli trial for whether edge (i->j) exists
            if random.random() < p_ij:
                G.add_edge(i, j)
    
    return G


def directed_holme_kim_graph(
    n,
    alpha=0.41,
    beta=0.54,
    gamma=0.05,
    delta_in=0.2,
    delta_out=0.0,
    p=0.3,
    seed=None,
    initial_graph=None,
):
    """
    Returns a directed graph grown by a Holme–Kim–style process combined
    with the alpha/beta/gamma preferential attachment approach from
    B. Bollobás, C. Borgs, J. Chayes, and O. Riordan, Directed scale-free graphs, Proceedings of the fourteenth annual ACM-SIAM Symposium on Discrete Algorithms, 132–139, 2003.
    *Poorly understood model made up by Maximilian Noichl*


    Parameters
    ----------
    n : int
        Desired number of nodes in the final graph.
    alpha : float
        Probability of 'alpha' event: add a new node v -> existing node w.
    beta : float
        Probability of 'beta' event: connect existing v -> existing w.
    gamma : float
        Probability of 'gamma' event: add a new node w, connected from existing v.
    delta_in : float
        Bias for choosing w by in-degree. If delta_in > 0, we sometimes pick w
        uniformly among all nodes instead of strictly in-degree-weighted.
    delta_out : float
        Bias for choosing v by out-degree. If delta_out > 0, we sometimes pick v
        uniformly among all nodes instead of strictly out-degree-weighted.
    p : float
        Probability to perform a "directed triangle closure" step after each edge.
        Interpreted as: if we add v->w, we may also add v->x if w->x (with prob p).
    seed : random_state or int
        Random number generator or seed.
    initial_graph : nx.MultiDiGraph, optional
        Starting graph. Otherwise, we begin with a small directed cycle of 3 nodes.

    Returns
    -------
    G : nx.DiGraph
        A directed graph grown via this hybrid mechanism.
    """
    rng = np.random.RandomState(seed)  # Using numpy's random number generator

    def _choose_node(candidates, node_list, delta):
        """Choose a node from 'candidates' (list repeated by degree),
        but with probability proportional to delta, choose uniformly
        from all of node_list.
        """
        if delta > 0:
            # Probability of uniform choice vs. degree-based choice
            bias_sum = len(node_list) * delta
            p_delta = bias_sum / (bias_sum + len(candidates))
            if rng.random() < p_delta:
                return rng.choice(node_list)
        return rng.choice(candidates)

    # --- 1. Initialize the Graph ---
    if initial_graph is not None:
        if not isinstance(initial_graph, nx.MultiDiGraph):
            raise nx.NetworkXError("initial_graph must be a MultiDiGraph (directed).")
        G = nx.DiGraph(initial_graph)  # or keep it MultiDiGraph if you prefer
    else:
        # Start with small seed (3-cycle, as in scale_free_graph)
        G = nx.DiGraph([(0,1), (1,2), (2,0)])

    if abs(alpha + beta + gamma - 1.0) > 1e-9:
        raise ValueError("alpha + beta + gamma must sum to 1.")

    # Precompute repeated-node lists for out-degree (vs) and in-degree (ws)
    vs = sum((count * [idx] for idx, count in G.out_degree()), [])
    ws = sum((count * [idx] for idx, count in G.in_degree()), [])

    # Also track all node labels
    node_list = list(G.nodes())

    # If existing nodes are numeric, start from the max index + 1
    numeric_nodes = [n for n in node_list if isinstance(n, numbers.Number)]
    if numeric_nodes:
        cursor = max(int(n) for n in numeric_nodes) + 1
    else:
        cursor = 3  # we started with nodes 0,1,2

    # --- 2. Growth Loop ---
    while len(G) < n:
        r = rng.random()
        if r < alpha:
            # ALPHA: new node (v), edge v->w
            v = cursor
            cursor += 1
            node_list.append(v)
            # choose w by in-degree preference (ws)
            w = _choose_node(ws, node_list, delta_in)
            G.add_edge(v, w)

            # update repeated-node lists
            vs.append(v)  # out-degree of v increased
            ws.append(w)  # in-degree of w increased

            # Holme-Kim "triangle closure": with probability p, if w->x, add v->x
            if rng.random() < p:
                # pick any out-neighbor x of w
                out_neighbors_w = list(G.successors(w))
                if out_neighbors_w:
                    x = rng.choice(out_neighbors_w)
                    if not G.has_edge(v, x):
                        G.add_edge(v, x)
                        vs.append(v)
                        ws.append(x)

        elif r < alpha + beta:
            # BETA: connect existing v->w
            v = _choose_node(vs, node_list, delta_out)
            w = _choose_node(ws, node_list, delta_in)
            G.add_edge(v, w)

            # update repeated-node lists
            vs.append(v)
            ws.append(w)

            # Holme-Kim closure step
            if rng.random() < p:
                out_neighbors_w = list(G.successors(w))
                if out_neighbors_w:
                    x = rng.choice(out_neighbors_w)
                    if not G.has_edge(v, x):
                        G.add_edge(v, x)
                        vs.append(v)
                        ws.append(x)

        else:
            # GAMMA: new node (w), connect v->w
            w = cursor
            cursor += 1
            node_list.append(w)
            # choose v by out-degree preference (vs)
            v = _choose_node(vs, node_list, delta_out)
            G.add_edge(v, w)

            # update repeated-node lists
            vs.append(v)
            ws.append(w)

            # Holme-Kim closure
            if rng.random() < p:
                out_neighbors_w = list(G.successors(w))
                if out_neighbors_w:
                    x = rng.choice(out_neighbors_w)
                    if not G.has_edge(v, x):
                        G.add_edge(v, x)
                        vs.append(v)
                        ws.append(x)

    return G




def generate_directed_s1_networkx(
    n=None,
    beta=-10,
    mu=-10,
    nu=-10,
    seed=None,
    # -- Hidden variable generation parameters --
    kappa_min=5.0,
    kappa_max=100.0,
    gamma=2.5,
    # -- Optionally supply your own in_kappa/out_kappa/angles --
    in_kappa=None,
    out_kappa=None,
    theta=None,
    # -- Optionally supply vertex names --
    num2name=None,
    # -- If True, we do out_kappa as a shuffled copy of in_kappa (similar to examples)
    shuffle_out_kappa=True,
    return_mu_and_theta=False
):
    """
    Generate a directed NetworkX DiGraph in the S1 model.

    This function optionally generates kappa_in/kappa_out from a Pareto-like 
    distribution truncated at kappa_max, as well as random angles theta. 
    Then it assigns edges among all vertex pairs based on probabilities 
    from the S1 model with parameters beta, mu, and nu.

    Parameters
    ----------
    n : int or None
        Number of vertices. Required if in_kappa and out_kappa are not provided.
    beta : float
        Clustering parameter (BETA). Must be set to a valid value (not -10).
    mu : float
        Average degree parameter (MU). If set to -10, the function will estimate
        it from the eventual in_kappa/out_kappa after generation.
    nu : float
        Reciprocity parameter (NU). Must be set to a valid value (not -10).
    seed : int, optional
        Random seed for reproducibility.
    kappa_min : float
        Minimum kappa for truncated Pareto generation.
    kappa_max : float
        Maximum kappa for truncated Pareto generation.
    gamma : float
        Pareto shape parameter.
    in_kappa : array-like or None
        If provided, these inKappa values are used directly. Otherwise,
        new ones are generated.
    out_kappa : array-like or None
        If provided, these outKappa values are used directly. Otherwise,
        new ones are generated (optional shuffling from in_kappa).
    theta : array-like or None
        Angles for each vertex. If None, random angles [0, 2π] are generated.
    num2name : list[str], optional
        If provided, the names for each vertex. Otherwise, vertices are named
        "v0", "v1", ..., "vN-1".
    shuffle_out_kappa : bool
        If True, and out_kappa is not provided, out_kappa is a random shuffle
        of in_kappa (simulating the “shuffle the kappa” step in the sample code).

    Returns
    -------
    G : nx.DiGraph
        The directed synthetic S1 network.
    final_theta : list[float]
        The (possibly generated) angles for each vertex.
    final_mu : float
        The final value of mu used in probability formulas.
    """

    # -------------------------------------------------------------------------
    # 1. Set random seeds and determine n, the number of vertices
    # -------------------------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if in_kappa is not None and out_kappa is not None:
        # Use the user-supplied inKappa/outKappa
        if len(in_kappa) != len(out_kappa):
            raise ValueError("in_kappa and out_kappa must be the same length.")
        n = len(in_kappa)
    else:
        # Must specify n if no kappa arrays given
        if n is None:
            raise ValueError("Please provide either `n` or both `in_kappa` and `out_kappa`.")

    # -------------------------------------------------------------------------
    # 2. Generate in_kappa/out_kappa if needed
    #    (with truncated Pareto distribution as in the example code)
    # -------------------------------------------------------------------------
    if in_kappa is None:
        # Generate a temporary list from Pareto, truncating at kappa_max
        # See the approach in "generate_hidden_variables_for_reciprocity_validation.py"
        in_kappa_vals = []
        needed = n
        while len(in_kappa_vals) < n:
            sample = (kappa_min * st.pareto.rvs(gamma - 1, size=needed))
            # Keep only the ones below kappa_max
            filtered = [v for v in sample if v < kappa_max]
            in_kappa_vals += filtered
            needed = n - len(in_kappa_vals)

        in_kappa_vals = in_kappa_vals[:n]  # ensure exactly n
        in_kappa_vals = np.array(in_kappa_vals)
        np.random.shuffle(in_kappa_vals)
        in_kappa = in_kappa_vals

    if out_kappa is None:
        # If we want to shuffle, do it. Otherwise we generate with the same distribution.
        if shuffle_out_kappa:
            out_kappa_vals = np.copy(in_kappa)
            np.random.shuffle(out_kappa_vals)
            out_kappa = out_kappa_vals
        else:
            # Generate new from the same distribution
            out_kappa_vals = []
            needed = n
            while len(out_kappa_vals) < n:
                sample = (kappa_min * st.pareto.rvs(gamma - 1, size=needed))
                filtered = [v for v in sample if v < kappa_max]
                out_kappa_vals += filtered
                needed = n - len(out_kappa_vals)

            out_kappa_vals = out_kappa_vals[:n]
            out_kappa_vals = np.array(out_kappa_vals)
            np.random.shuffle(out_kappa_vals)
            out_kappa = out_kappa_vals

    in_kappa = np.array(in_kappa)
    out_kappa = np.array(out_kappa)

    # -------------------------------------------------------------------------
    # 3. Check beta/nu, and compute mu if needed
    # -------------------------------------------------------------------------
    if beta == -10:
        raise ValueError("ERROR: The value of parameter beta must be provided (beta != -10).")
    if nu == -10:
        raise ValueError("ERROR: The value of parameter nu must be provided (nu != -10).")

    # If mu == -10, compute from the average kappa as in the C++ code
    if mu == -10:
        avg_kappa = 0.0
        for v in range(n):
            avg_kappa += (in_kappa[v] + out_kappa[v]) / (2.0 * n)
        mu = beta * math.sin(math.pi / beta) / (2.0 * math.pi * avg_kappa)

    # -------------------------------------------------------------------------
    # 4. Generate angles if not provided
    # -------------------------------------------------------------------------
    if theta is None:
        theta = [random.uniform(0, 2.0 * math.pi) for _ in range(n)]
    else:
        if len(theta) != n:
            raise ValueError("theta list must match the number of vertices.")

    # -------------------------------------------------------------------------
    # 5. Create a directed graph, add all vertices with attributes
    # -------------------------------------------------------------------------
    G = nx.DiGraph()
    if num2name is None:
        num2name = [f"v{i}" for i in range(n)]

    for i in range(n):
        G.add_node(
            num2name[i],
            in_kappa=float(in_kappa[i]),
            out_kappa=float(out_kappa[i]),
            theta=float(theta[i])
        )

    # -------------------------------------------------------------------------
    # 6. Determine edges using S1 probabilities
    # -------------------------------------------------------------------------
    numerical_zero = 1e-5
    prefactor = n / (2.0 * math.pi * mu)

    for v1 in range(n):
        kout1 = out_kappa[v1]
        kin1 = in_kappa[v1]
        theta1 = theta[v1]
        for v2 in range(v1 + 1, n):
            theta2 = theta[v2]
            # Circular distance
            dtheta = math.pi - abs(math.pi - abs(theta1 - theta2))

            # Probability p12 (v1 -> v2)
            koutkin = kout1 * in_kappa[v2]
            if koutkin > numerical_zero:
                p12 = 1.0 / (1.0 + ((prefactor * dtheta) / koutkin) ** beta)
            else:
                p12 = 0.0

            # Probability p21 (v2 -> v1)
            koutkin = out_kappa[v2] * kin1
            if koutkin > numerical_zero:
                p21 = 1.0 / (1.0 + ((prefactor * dtheta) / koutkin) ** beta)
            else:
                p21 = 0.0

            # Combine probabilities with reciprocity factor nu
            if nu > 0:
                if p12 < p21:
                    p11 = ((1.0 - nu) * p21 + nu) * p12
                else:
                    p11 = ((1.0 - nu) * p12 + nu) * p21
            else:
                # p12 + p21 < 1 check
                if p12 + p21 < 1.0:
                    p11 = (1.0 + nu) * p12 * p21
                else:
                    p11 = (1.0 + nu) * p12 * p21 + nu * (1.0 - p12 - p21)

            # Randomly determine which edges to add
            r = random.random()

            if r < p11:
                # Bidirectional
                G.add_edge(num2name[v1], num2name[v2])
                G.add_edge(num2name[v2], num2name[v1])
            elif r < p21:
                # Only v2 -> v1
                G.add_edge(num2name[v2], num2name[v1])
            elif r < (p21 + p12 - p11):
                # Only v1 -> v2
                G.add_edge(num2name[v1], num2name[v2])
            # else: no edge
    if return_mu_and_theta:
        return G, theta, mu
    else:
        return G
