import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter


class BernoulliRandomGraph:
    def __init__(self, N, K):
        """
        Random binary graph class.
        :param N: Number of nodes.
        :param K: Number of blocks.
        """
        # Hyperparameters.
        self._N = N
        self._K = K
        self._a = 1 / 2
        self._b = 1 / 2
        self._alpha = 1 / 2

        # Parameters and latent variables.
        self._theta = np.zeros((self._K, self._K))
        self._pi = np.zeros(self._K)
        self._Z = np.zeros(self._N)

        # Random graph sample.
        self._X = np.zeros((self._N, self._N))
        
    def generate(self, nx_graph):
        """
        Generates a random binary graph.
        :return:
        """
        # Parameters and latent variables.
        self._theta = np.zeros((self._K, self._K))
        self._pi = np.random.dirichlet([self._alpha] * self._K)
        self._Z = np.random.choice(self._K, self._N, p=self._pi)
        for i in range(self._K):
            self._theta[i, i:] = np.random.beta(self._a, self._b, self._K - i)
        self._theta = np.maximum(self._theta, self._theta.T)
        for i in range(self._N):
            self._X[i, i + 1:] = np.random.binomial(1, self._theta[self._Z[i], self._Z[i + 1:]])
        self._X = np.maximum(self._X, self._X.T)
        if nx_graph:
            return nx.from_numpy_array(self._X), self._Z
        return self._X, self._Z

    def visualize(self):
        """
        Visualizing the block structured matrix.
        :return:
        """
        block_sizes = Counter(self._Z)
        new_order = list(chain(*[np.where(self._Z == k)[0] for k in list(block_sizes.keys())]))
        X_blocked = self._X[:, new_order][new_order]
        fig, ax = plt.subplots(1, 1)
        for l in np.cumsum(list(block_sizes.values())):
            ax.axhline(l, 0, self._N, color='black')
            ax.axvline(l, 0, self._N, color='black')
        ax.imshow(X_blocked, origin='lower')
        plt.show()
