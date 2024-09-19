import random
import operator
import numpy as np
import networkx as nx
from math import lgamma
from scipy.special import digamma


def train_test_split(G, perc_edges=10):
    """
    Set aside a percentage of edges and a percentage
    of non-edges in the graph. Convert to CSR format.
    :param G: nx.Graph()
    :param perc_edges: Percentage of edges to set aside.
    :return:
    """
    #num_vertices = len(G)

    # Sample edges.
    number_sampled_edges = int((perc_edges / 100) * nx.number_of_edges(G))
    sampled_edges = random.sample(list(G.edges()), number_sampled_edges)

    '''
    # Sample nonedges.
    sampled_nonedges = []
    number_sampled_nonedges = 0
    while number_sampled_nonedges < number_sampled_edges:
        i, j = np.random.randint(0, num_vertices, 2)
        if i != j and not G.has_edge(i, j) and (i, j) not in sampled_nonedges \
                and (j, i) not in sampled_nonedges:
            sampled_nonedges.append((i, j))
            number_sampled_nonedges += 1
    '''
    # Remove sampled edges.
    G.remove_edges_from(sampled_edges)

    # Convert to CSR.
    return nx.to_scipy_sparse_matrix(G), sampled_edges




