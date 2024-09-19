import numpy as np
from math import lgamma
import time
from scipy.special import digamma

vectorized_lgamma = np.vectorize(lgamma)
vectorized_digamma = np.vectorize(digamma)


class BernoulliModelState:
    def __init__(self, X, K, sampled_edges):
        """
        Mean-field variational inference for the Bernoulli Stochastic Blockmodel.
        :param X: scipy.sparse.csr_matrix sparse matrix representing the graph.
        :param K: Number of clusters. If K=None, X is fitted with a Bayesian
        Nonparametric prior on the block proportions.
        :param sampled_edges: Edges that are not used in the training dataset.
        """
        self._X = X
        self._K = K
        self._sampled_from = [edge[0] for edge in sampled_edges]
        self._sampled_to = [edge[1] for edge in sampled_edges]
        self._number_sampled_edges = len(sampled_edges)

        # Number of nodes.
        self._N = X.shape[0]

        # Variational parameters.
        self._alpha = np.zeros(shape=K)
        self._tau = np.zeros(shape=(self._N, K))
        self._a = np.zeros(shape=(K, K))
        self._b = np.zeros(shape=(K, K))

        # Initialization of the variational parameters.
        self._initialize()

        # Store newly computed tau values.
        self._tau_new = np.zeros(shape=(self._N, K))

    def _local_update(self):
        """
        Vectorized implementation of the local update equation.
        :return: None
        """
        expectation = vectorized_digamma(self._b) - vectorized_digamma(self._a + self._b)
        self._tau_new[:] = (self._tau.sum(axis=0) - self._tau) @ expectation
        expectation = vectorized_digamma(self._a) - vectorized_digamma(self._b)
        self._tau_new += self._X @ self._tau @ expectation  # Slow
        self._tau_new += vectorized_digamma(self._alpha) - digamma(self._alpha.sum())

    def _normalize_tau(self):
        """
        Exponentiate log values and normalize.
        :return: None
        """
        self._tau_new[:] = np.exp(self._tau_new - np.amax(self._tau_new, axis=1)[:, np.newaxis])  # Slow
        self._tau[:] = self._tau_new / self._tau_new.sum(axis=1)[:, np.newaxis]

    def _global_update(self):
        """
        Vectorized implementation of the global update equation.
        :return: None
        """
        # Dirichlet parameter update.
        tau_sum = self._tau.sum(axis=0)
        self._alpha[:] = tau_sum

        # Beta parameter update for a.
        self._a[:] = self._tau.T @ self._X @ self._tau  # Slow

        # Beta parameter update for b.
        self._b[:] = np.kron(tau_sum, tau_sum).reshape(self._K, self._K)
        self._b -= (self._tau.T @ self._tau + self._a)

        # Divide diagonals by 2 as we have summed the (k, k) entries twice.
        np.fill_diagonal(self._a, self._a.diagonal() / 2)
        np.fill_diagonal(self._b, self._b.diagonal() / 2)

        # Add hyperparameters.
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2

    def _initialize(self):
        """
        Random initialization of the variational parameters.
        :return: None
        """
        print('Initialization...')
        for i in range(self._N):
            self._tau[i, np.random.randint(0, self._K)] = 1
        self._global_update()

    def update(self, normalize=True):
        """
        Update model state.
        :param normalize: Flag value, if True then the variables are updated as in standard
        VI, if False then the contribution of the weighed model is added to the update.
        :return: None
        """
        self._global_update()
        self._local_update()
        if normalize:
            self._normalize_tau()

    def compute_perplexity(self):
        """
        Average log-predictive on the test set.
        :return: float
        """
        return np.exp(-np.sum((self._tau[self._sampled_from].T @ self._tau[self._sampled_to]) *
                              (vectorized_digamma(self._a) - vectorized_digamma(self._a + self._b))) /
                      self._number_sampled_edges)


class BernoulliModelStochasticState:
    def __init__(self, X, K, S):
        """
        Mean-field stochastic variational inference for the Bernoulli Stochastic Blockmodel.
        :param X: scipy.sparse.csr_matrix sparse matrix representing the graph.
        :param K: Number of blocks.
        :param S: Number of nodes to subsample.
        """
        self._X = X
        self._K = K
        self._S = S

        # Number of nodes.
        self._N = X.shape[0]

        # Variational parameters.
        self._tau = np.zeros(shape=(self._N, K))
        self._alpha = np.zeros(shape=K)
        self._a = np.zeros(shape=(K, K))
        self._b = np.zeros(shape=(K, K))

        # Natural gradients of the variational parameters.
        self._alpha_grad = np.zeros(shape=K)
        self._a_grad = np.zeros(shape=(K, K))
        self._b_grad = np.zeros(shape=(K, K))

        # Initialize variational parameters.
        self._initialize()

        # List of sampled nodes and a mask for sampled indices.
        self._sampled_nodes = np.random.choice(a=self._N, size=S, replace=False)
        self._sampled_mask = np.full(shape=self._N, fill_value=False, dtype=bool)
        self._sampled_mask[self._sampled_nodes] = True

        # Store new tau values.
        self._tau_new = np.zeros(shape=(self._S, self._K))

        # Normalizing factors for the stochastic gradients.
        self._norm_factor = np.zeros(shape=(K, K))
        self._norm_factor[~np.eye(K, dtype=bool)] = (self._N * (self._N - 1)) / (self._S * (2 * self._N - self._S - 1))
        np.fill_diagonal(self._norm_factor, (self._N * (self._N - 1) / 2) /
                         (self._S * (self._S - 1) / 2 + self._S * (self._N - self._S)))

        # Lower triangle indices of the inter- and intra-block parameters.
        self.tril_inds = np.tril_indices(K, - 1)

    def _initialize(self):
        for i in range(self._N):
            self._tau[i, np.random.randint(0, self._K)] = 1
        tau_sum = self._tau.sum(axis=0)
        self._alpha = tau_sum
        self._a = self._tau.T @ self._X @ self._tau
        self._b = np.kron(tau_sum, tau_sum).reshape(self._K, self._K)
        self._b -= (self._tau.T @ self._tau + self._a)
        np.fill_diagonal(self._a, self._a.diagonal() / 2)
        np.fill_diagonal(self._b, self._b.diagonal() / 2)
        self._alpha += 1 / 2
        self._a += 1 / 2
        self._b += 1 / 2

    def _local_update(self):
        expectation = vectorized_digamma(self._b) - vectorized_digamma(self._a + self._b)
        self._tau_new[:] = (self._tau.sum(axis=0) - self._tau[self._sampled_nodes]) @ expectation
        expectation = vectorized_digamma(self._a) - vectorized_digamma(self._b)
        self._tau_new += self._X[self._sampled_nodes] @ self._tau @ expectation
        self._tau_new += vectorized_digamma(self._alpha) - digamma(self._alpha.sum())
        self._tau_new = np.exp(self._tau_new - np.amax(self._tau_new, axis=1)[:, np.newaxis])
        self._tau_new = self._tau_new / self._tau_new.sum(axis=1)[:, np.newaxis]
        self._tau[self._sampled_nodes] = self._tau_new

    def _global_update(self, rho):
        self._alpha_grad += (self._N / self._S) * self._tau[self._sampled_nodes].sum(axis=0)
        half_tau = self._tau * (~self._sampled_mask + self._sampled_mask / 2)[:, np.newaxis]
        aux = self._tau[self._sampled_nodes].T @ self._X[self._sampled_nodes] @ half_tau
        self._a_grad += np.triu(aux) + np.tril(aux, k=-1).T
        aux = np.kron(self._tau[self._sampled_nodes].sum(axis=0), half_tau.sum(axis=0)).reshape(self._K, self._K) - \
              (self._tau[self._sampled_nodes].T @ self._tau[self._sampled_nodes]) / 2
        self._b_grad += np.triu(m=aux) + np.tril(m=aux, k=-1).T
        self._b_grad -= self._a_grad
        self._a_grad *= self._norm_factor
        self._b_grad *= self._norm_factor
        self._a_grad[self.tril_inds] = self._a_grad.T[self.tril_inds]
        self._b_grad[self.tril_inds] = self._b_grad.T[self.tril_inds]        
        self._a_grad += 1 / 2
        self._b_grad += 1 / 2
        self._alpha_grad += 1 / 2
        self._alpha = (1 - rho) * self._alpha + rho * self._alpha_grad
        self._a = (1 - rho) * self._a + rho * self._a_grad
        self._b = (1 - rho) * self._b + rho * self._b_grad

    def update_state(self, rho):
        self._local_update()
        self._global_update(rho)        
        self._sample()

    def _sample(self):
        self._sampled_nodes = np.random.choice(a=self._N, size=self._S, replace=False)
        self._sampled_mask.fill(False)
        self._sampled_mask[self._sampled_nodes] = True
        self._alpha_grad.fill(0.0)
        self._a_grad.fill(0.0)
        self._b_grad.fill(0.0)