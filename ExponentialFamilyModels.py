import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from scipy.special import factorial, log1p
from math import lgamma
from scipy.special import digamma

vectorized_lgamma = np.vectorize(lgamma)
vectorized_digamma = np.vectorize(digamma)


class WeightedModelState(ABC):
    def __init__(self, W, K, sampled_edges):
        """
        Mean-field variational inference for the Weighted Stochastic Blockmodel.
        :param W: scipy.sparse.csr_matrix sparse matrix representing the weighted network.
        :param K: Number of blocks.
        :param sampled_edges: Edges that are not used in the training dataset.
        """
        super().__init__()
        self._W = W
        self._K = K
        self._sampled_from = [edge[0] for edge in sampled_edges]
        self._sampled_to = [edge[1] for edge in sampled_edges]
        self._number_sampled_edges = len(sampled_edges)

        # Number of nodes.
        self._N = W.shape[0]

        # Binarize the weighted matrix to efficiently update the parameters
        # of the underlying Bernoulli model.
        self._X = self._W.copy()
        self._X.data[:] = 1

        # Variational free parameters.
        self._tau = np.zeros(shape=(self._N, self._K))
        for i in range(self._N):
            self._tau[i, np.random.randint(0, self._K)] = 1
        self._alpha = np.zeros(shape=self._K)

        # Bernoulli-specific model parameters.
        self._a = np.zeros(shape=(self._K, self._K))
        self._b = np.zeros(shape=(self._K, self._K))
        self._update_bernoulli_global_params()

        # Exponential family-specific model parameters.
        self._nu = np.zeros(shape=(self._K, self._K))
        self._lambda = None

        # Instantiated beforehand to
        self._tau_new = np.zeros(shape=(self._N, self._K))

    def _update_bernoulli_local_params(self):
        expectation = vectorized_digamma(self._b) - vectorized_digamma(self._a + self._b)
        self._tau_new[:] = (np.sum(self._tau, axis=0) - self._tau) @ expectation
        expectation = vectorized_digamma(self._a) - vectorized_digamma(self._b)
        self._tau_new += self._X @ self._tau @ expectation
        self._tau_new += vectorized_digamma(self._alpha) - digamma(np.sum(self._alpha))

    def _update_bernoulli_global_params(self):
        tau_sum = np.sum(self._tau, axis=0)
        self._alpha = tau_sum
        self._a = self._tau.T @ self._X @ self._tau
        self._b = np.kron(tau_sum, tau_sum).reshape(self._K, self._K)
        self._b -= (self._tau.T @ self._tau + self._a)
        np.fill_diagonal(self._a, self._a.diagonal() / 2)
        np.fill_diagonal(self._b, self._b.diagonal() / 2)

    def _normalize_and_update_local_params(self):
        self._tau_new = np.exp(self._tau_new - np.amax(self._tau_new, axis=1)[:, np.newaxis])
        self._tau_new = self._tau_new / self._tau_new.sum(axis=1)[:, np.newaxis]
        self._tau[:] = self._tau_new

    @abstractmethod
    def _update_weighted_local_params(self):
        pass

    @abstractmethod
    def _update_weighted_global_params(self):
        pass

    @abstractmethod
    def _expected_natural_parameter(self, lambda_, nu):
        pass

    @abstractmethod
    def _expected_normalizing_constant(self, lambda_, nu):
        pass

    @abstractmethod
    def update(self):
        pass


class PoissonModelState(WeightedModelState):
    def __init__(self, W, K):
        WeightedModelState.__init__(self, W, K)
        self._nu0 = 0.001
        self._lambda0 = -0.999
        self._T = W
        self._log_h = csr_matrix((-log1p(factorial(W.data)), W.indices, W.indptr))
        self._lambda = np.zeros(shape=(K, K))
        self._compute_weighted_global_params()

    def _compute_weighted_local_params(self):
        self._tau_new += np.sum(self._log_h @ self._tau, axis=1)[:, np.newaxis]
        expectation = self._expected_natural_parameter(self._lambda, self._nu)
        self._tau_new += self._T @ self._tau @ expectation
        expectation = self._expected_normalizing_constant(self._lambda, self._nu)
        self._tau_new += self._X @ self._tau @ expectation

    def _compute_weighted_global_params(self):
        self._nu[:] = self._a
        self._lambda = self._tau.T @ self._T @ self._tau
        np.fill_diagonal(self._lambda, self._lambda.diagonal() / 2)

    def _expected_natural_parameter(self, lambda_, nu):
        return vectorized_digamma(lambda_ + 1) - np.log(nu)

    def _expected_normalizing_constant(self, lambda_, nu):
        return -(lambda_ + 1) / nu

    def update(self):
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2
        self._lambda += self._lambda0
        self._nu += self._nu0
        self._update_bernoulli_local_params()
        self._compute_weighted_local_params()
        self._normalize_and_update_local_params()
        self._update_bernoulli_global_params()
        self._compute_weighted_global_params()


class ExponentialModelState(WeightedModelState):
    def __init__(self, W, K):
        WeightedModelState.__init__(self, W, K)
        self._nu0 = -0.999
        self._lambda0 = 0.001
        self._T = W
        self._log_h = 0
        self._lambda = np.zeros(shape=(K, K))
        self._compute_weighted_global_params()

    def _compute_weighted_local_params(self):
        expectation = self._expected_natural_parameter(self._lambda, self._nu)
        self._tau_new += self._T @ self._tau @ expectation
        expectation = self._expected_normalizing_constant(self._lambda, self._nu)
        self._tau_new += self._X @ self._tau @ expectation

    def _compute_weighted_global_params(self):
        self._nu[:] = self._a
        self._lambda = self._tau.T @ self._T @ self._tau
        np.fill_diagonal(self._lambda, self._lambda.diagonal() / 2)

    def _expected_natural_parameter(self, lambda_, nu):
        return -(nu + 1) / lambda_

    def _expected_normalizing_constant(self, lambda_, nu):
        return vectorized_digamma(nu + 1) - np.log(lambda_)

    def update(self):
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2
        self._lambda += self._lambda0
        self._nu += self._nu0
        self._compute_bernoulli_local_params()
        self._compute_weighted_local_params()
        self._normalize_and_update_local_params()
        self._compute_bernoulli_global_params()
        self._compute_weighted_global_params()


class GaussianModelState(WeightedModelState):
    def __init__(self, W, K):
        WeightedModelState.__init__(self, W, K)
        self._nu0 = 1
        self._lambda01 = 0
        self._lambda02 = 2
        self._T = W, W.power(2)
        self._log_h = -np.log(np.sqrt(2 * np.pi))
        self._lambda = np.zeros(shape=(2, K, K))
        self._compute_weighted_global_params()

    def _compute_weighted_local_params(self):
        self._tau_new += self._log_h * np.sum(self._X @ self._tau, axis=1)[:, np.newaxis]
        expectation = self._expected_natural_parameter(self._lambda, self._nu)
        self._tau_new += self._T[0] @ self._tau @ expectation[0]
        self._tau_new += self._T[1] @ self._tau @ expectation[1]
        expectation = self._expected_normalizing_constant(self._lambda, self._nu)
        self._tau_new += self._X @ self._tau @ expectation

    def _compute_weighted_global_params(self):
        self._nu[:] = self._a
        self._lambda[0] = self._tau.T @ self._T[0] @ self._tau
        self._lambda[1] = self._tau.T @ self._T[1] @ self._tau
        np.fill_diagonal(self._lambda[0], self._lambda[0].diagonal() / 2)
        np.fill_diagonal(self._lambda[1], self._lambda[1].diagonal() / 2)

    def _expected_natural_parameter(self, lambda_, nu):
        return ((nu + 1) * lambda_[0]) / (lambda_[1] * nu - lambda_[0] ** 2), \
               -(nu * (nu + 1)) / (2 * (lambda_[1] * nu - lambda_[0] ** 2))

    def _expected_normalizing_constant(self, lambda_, nu):
        x = lambda_[0]
        y = lambda_[1]
        z = nu
        return - ((x ** 2) * (np.log((y * z - x ** 2) / (2 * z)) - vectorized_digamma((z + 1) / 2) - 1) +
                  y * (z * (vectorized_digamma((z + 1) / 2) - np.log((y * z - x ** 2) / (2 * z))) - 1)) / \
               (2 * (x ** 2 - y * z))

        #return - (1 / 2) * ((lambda_[1] + lambda_[0] ** 2) / (lambda_[1] * nu - lambda_[0] ** 2) -
        #                    vectorized_digamma((nu + 1) / 2) -
        #                    np.log((2 * nu) / (lambda_[1] * nu - lambda_[0] ** 2)))

    def update(self):
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2
        self._lambda[0] += self._lambda01
        self._lambda[1] += self._lambda02
        self._nu += self._nu0
        self._compute_bernoulli_local_params()
        self._compute_weighted_local_params()
        self._normalize_and_update_local_params()
        self._compute_bernoulli_global_params()
        self._compute_weighted_global_params()


class WeightedModelStochasticState(ABC):
    def __init__(self, W, K, S):
        super().__init__()
        self._W = W
        self._X = W.copy()
        self._X.data[:] = 1
        self._N = W.shape[0]
        self._K = K
        self._S = S

        # Variational free parameters.
        self._tau = np.zeros(shape=(self._N, K))
        for i in range(self._N):
            self._tau[i, np.random.randint(0, self._K)] = 1
        self._alpha = np.zeros(shape=K)        
        self._a = np.zeros(shape=(K, K))
        self._b = np.zeros(shape=(K, K))
        self._alpha_grad = np.zeros(shape=K)
        self._a_grad = np.zeros(shape=(K, K))        
        self._b_grad = np.zeros(shape=(K, K))
        self._initialize_bernoulli_model()
        self._nu = np.zeros(shape=(K, K))
        self._nu_grad = np.zeros(shape=(K, K))
        self._lambda = None
        self._lambda_grad = None

        self._sampled_nodes = np.random.choice(a=self._N, size=S, replace=False)
        self._sampled_mask = np.full(shape=self._N, fill_value=False, dtype=bool)
        self._sampled_mask[self._sampled_nodes] = True
        self._tau_new = np.zeros(shape=(self._S, self._K))
        self.norm_factor = np.zeros(shape=(K, K))
        for k in range(K):
            for l in range(K):
                if k == l:
                    self.norm_factor[k, l] = (self._N * (self._N - 1) / 2) / \
                                             (self._S * (self._S - 1) / 2 +
                                              self._S * (self._N - self._S))
                else:
                    self.norm_factor[k, l] = (self._N * (self._N - 1)) / \
                                             (self._S * (2 * self._N - self._S - 1))

        # Indexing for the lower triangle.
        self.tril_inds = np.tril_indices(K, - 1)

    def _compute_bernoulli_local_params(self):
        expectation = vectorized_digamma(self._b) - vectorized_digamma(self._a + self._b)
        self._tau_new[:] = (np.sum(self._tau, axis=0) - self._tau[self._sampled_nodes, :]) @ expectation
        expectation = vectorized_digamma(self._a) - vectorized_digamma(self._b)
        self._tau_new += self._X[self._sampled_nodes] @ self._tau @ expectation
        self._tau_new += vectorized_digamma(self._alpha) - scalar_digamma(np.sum(self._alpha))

    def _compute_bernoulli_natural_grads(self):
        self._alpha_grad += (self._N / self._S) * np.sum(self._tau[self._sampled_nodes], axis=0)
        half_tau = self._tau * (~self._sampled_mask + self._sampled_mask / 2)[:, np.newaxis]
        aux = self._tau[self._sampled_nodes].T @ self._X[self._sampled_nodes] @ half_tau
        self._a_grad += np.triu(aux) + np.tril(aux, k=-1).T
        aux = np.kron(self._tau[self._sampled_nodes].sum(axis=0), half_tau.sum(axis=0)).reshape(self._K, self._K) - \
              (self._tau[self._sampled_nodes].T @ self._tau[self._sampled_nodes]) / 2
        self._b_grad += np.triu(m=aux) + np.tril(m=aux, k=-1).T
        self._b_grad -= self._a_grad
        self._a_grad *= self.norm_factor
        self._b_grad *= self.norm_factor
        self._a_grad[self.tril_inds] = self._a_grad.T[self.tril_inds]
        self._b_grad[self.tril_inds] = self._b_grad.T[self.tril_inds]
        return half_tau

    def _initialize_bernoulli_model(self):
        self._alpha = self._tau.sum(axis=0)
        self._a = self._tau.T @ self._X @ self._tau
        tau_sum = self._tau.sum(axis=0)
        self._b = np.kron(tau_sum, tau_sum).reshape(self._K, self._K)
        self._b -= (self._tau.T @ self._tau + self._a)
        np.fill_diagonal(self._a, self._a.diagonal() / 2)
        np.fill_diagonal(self._b, self._b.diagonal() / 2)

    def _update_bernoulli_model(self, rho):
        self._alpha = (1 - rho) * self._alpha + rho * self._alpha_grad
        self._a = (1 - rho) * self._a + rho * self._a_grad
        self._b = (1 - rho) * self._b + rho * self._b_grad        
        self._alpha_grad.fill(0.0)
        self._a_grad.fill(0.0)
        self._b_grad.fill(0.0)

    @abstractmethod
    def _compute_weighted_local_params(self):
        pass

    @abstractmethod
    def _compute_weighted_natural_grads(self, half_tau):
        pass

    @abstractmethod
    def _initialize_weighted_model(self):
        pass

    @abstractmethod
    def _update_weighted_model(self, rho):
        pass

    def _normalize_and_update_local_params(self):
        self._tau_new = np.exp(self._tau_new - np.amax(self._tau_new, axis=1)[:, np.newaxis])
        self._tau_new = self._tau_new / self._tau_new.sum(axis=1)[:, np.newaxis]
        self._tau[self._sampled_nodes] = self._tau_new

    @abstractmethod
    def _expected_natural_parameter(self, lambda_, nu):
        pass

    @abstractmethod
    def _expected_normalizing_constant(self, lambda_, nu):
        pass

    def _sample(self):
        self._sampled_nodes = np.random.choice(a=self._N, size=self._S, replace=False)
        self._sampled_mask.fill(False)
        self._sampled_mask[self._sampled_nodes] = True
        self._alpha_grad.fill(0.0)
        self._a_grad.fill(0.0)
        self._b_grad.fill(0.0)
        self._nu_grad.fill(0.0)

    @abstractmethod
    def update_state(self, rho):
        pass


class PoissonModelStochasticState(WeightedModelStochasticState):
    def __init__(self, X, K, S):
        WeightedModelStochasticState.__init__(self, X, K, S)
        self._nu0 = 0.001
        self._lambda0 = -0.999
        self._T = X
        self._log_h = csr_matrix((-log1p(factorial(X.data)), X.indices, X.indptr))
        self._lambda = np.zeros(shape=(K, K))
        self._lambda_grad = np.zeros(shape=(K, K))
        self._initialize_weighted_model()

    def _update_weighted_model(self, rho):
        self._nu = (1 - rho) * self._nu + rho * self._nu_grad
        self._lambda = (1 - rho) * self._lambda + rho * self._lambda_grad
        self._nu_grad.fill(0.0)
        self._lambda_grad.fill(0.0)

    def _compute_weighted_local_params(self):
        self._tau_new += np.sum(self._log_h[self._sampled_nodes] @ self._tau, axis=1)[:, np.newaxis]
        expectation = self._expected_natural_parameter(self._lambda, self._nu)
        self._tau_new += self._T[self._sampled_nodes] @ self._tau @ expectation
        expectation = self._expected_normalizing_constant(self._lambda, self._nu)
        self._tau_new += self._X[self._sampled_nodes] @ self._tau @ expectation

    def _compute_weighted_natural_grads(self, half_tau):
        self._nu_grad[:] = self._a_grad
        aux = self._tau[self._sampled_nodes].T @ self._T[self._sampled_nodes] @ half_tau
        self._lambda_grad += np.triu(aux) + np.tril(aux, k=-1).T
        self._lambda_grad *= self.norm_factor
        self._lambda_grad[self.tril_inds] = self._lambda_grad.T[self.tril_inds]

    def _expected_natural_parameter(self, lambda_, nu):
        return vectorized_digamma(lambda_ + 1) - np.log(nu)

    def _expected_normalizing_constant(self, lambda_, nu):
        return -(lambda_ + 1) / nu

    def _initialize_weighted_model(self):
        self._nu[:] = self._a
        self._lambda = self._tau.T @ self._T @ self._tau
        np.fill_diagonal(self._lambda, self._lambda.diagonal() / 2)
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2
        self._lambda += self._lambda0
        self._nu += self._nu0

    def _add_hyperparams_to_grads(self):
        self._a_grad += 1 / 2
        self._b_grad += 1 / 2
        self._alpha_grad += 1 / 2
        self._lambda_grad += self._lambda0
        self._nu_grad += self._nu0

    def update_state(self, rho):
        self._compute_bernoulli_local_params()
        self._compute_weighted_local_params()
        self._normalize_and_update_local_params()
        half_tau = self._compute_bernoulli_natural_grads()
        self._compute_weighted_natural_grads(half_tau)
        self._add_hyperparams_to_grads()
        self._update_bernoulli_model(rho)
        self._update_weighted_model(rho)
        self._sample()
        self._lambda_grad.fill(0.0)


class ExponentialModelStochasticState(WeightedModelStochasticState):
    def __init__(self, X, K, S):
        WeightedModelStochasticState.__init__(self, X, K, S)
        self._nu0 = -0.999
        self._lambda0 = 0.001
        self._T = X
        self._log_h = 0
        self._lambda = np.zeros(shape=(K, K))
        self._lambda_grad = np.zeros(shape=(K, K))
        self._initialize_weighted_model()

    def _update_weighted_model(self, rho):
        self._nu = (1 - rho) * self._nu + rho * self._nu_grad
        self._lambda = (1 - rho) * self._lambda + rho * self._lambda_grad
        self._nu_grad.fill(0.0)
        self._lambda_grad.fill(0.0)

    def _compute_weighted_local_params(self):
        expectation = self._expected_natural_parameter(self._lambda, self._nu)
        self._tau_new += self._T[self._sampled_nodes] @ self._tau @ expectation
        expectation = self._expected_normalizing_constant(self._lambda, self._nu)
        self._tau_new += self._X[self._sampled_nodes] @ self._tau @ expectation

    def _compute_weighted_natural_grads(self, half_tau):
        self._nu_grad[:] = self._a_grad
        aux = self._tau[self._sampled_nodes].T @ self._T[self._sampled_nodes] @ half_tau
        self._lambda_grad += np.triu(aux) + np.tril(aux, k=-1).T
        self._lambda_grad *= self.norm_factor
        self._lambda_grad[self.tril_inds] = self._lambda_grad.T[self.tril_inds]

    def _expected_natural_parameter(self, lambda_, nu):
        return -(nu + 1) / lambda_

    def _expected_normalizing_constant(self, lambda_, nu):
        return vectorized_digamma(nu + 1) - np.log(lambda_)

    def _initialize_weighted_model(self):
        self._nu[:] = self._a
        self._lambda = self._tau.T @ self._T @ self._tau
        np.fill_diagonal(self._lambda, self._lambda.diagonal() / 2)
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2
        self._lambda += self._lambda0
        self._nu += self._nu0

    def _add_hyperparams_to_grads(self):
        self._a_grad += 1 / 2
        self._b_grad += 1 / 2
        self._alpha_grad += 1 / 2
        self._lambda_grad += self._lambda0
        self._nu_grad += self._nu0

    def update_state(self, rho):
        self._compute_bernoulli_local_params()
        self._compute_weighted_local_params()
        self._normalize_and_update_local_params()
        half_tau = self._compute_bernoulli_natural_grads()
        self._compute_weighted_natural_grads(half_tau)
        self._add_hyperparams_to_grads()
        self._update_bernoulli_model(rho)
        self._update_weighted_model(rho)
        self._sample()
        self._lambda_grad.fill(0.0)


class GaussianModelStochasticState(WeightedModelStochasticState):
    def __init__(self, X, K, S):
        WeightedModelStochasticState.__init__(self, X, K, S)
        self._nu0 = 1
        self._lambda01 = 0
        self._lambda02 = 2
        self._T = X, X.power(2)
        self._log_h = -np.log(np.sqrt(2 * np.pi))
        self._lambda = np.zeros(shape=(2, K, K))
        self._lambda_grad = np.zeros(shape=(2, K, K))
        self._initialize_weighted_model()

    def _update_weighted_model(self, rho):
        self._nu = (1 - rho) * self._nu + rho * self._nu_grad
        self._lambda = (1 - rho) * self._lambda + rho * self._lambda_grad
        self._nu_grad.fill(0.0)
        self._lambda_grad.fill(0.0)

    def _compute_weighted_local_params(self):
        self._tau_new += np.sum(self._X[self._sampled_nodes] @ self._tau * self._log_h, axis=1)[:, np.newaxis]
        expectation = self._expected_natural_parameter(self._lambda, self._nu)
        self._tau_new += self._T[0][self._sampled_nodes] @ self._tau @ expectation[0]
        self._tau_new += self._T[1][self._sampled_nodes] @ self._tau @ expectation[1]
        expectation = self._expected_normalizing_constant(self._lambda, self._nu)
        self._tau_new += self._X[self._sampled_nodes] @ self._tau @ expectation

    def _compute_weighted_natural_grads(self, half_tau):
        self._nu_grad[:] = self._a_grad
        aux = self._tau[self._sampled_nodes].T @ self._T[0][self._sampled_nodes] @ half_tau
        self._lambda_grad[0] += np.triu(aux) + np.tril(aux, k=-1).T
        self._lambda_grad[0] *= self.norm_factor
        self._lambda_grad[0][self.tril_inds] = self._lambda_grad[0].T[self.tril_inds]
        aux = self._tau[self._sampled_nodes].T @ self._T[1][self._sampled_nodes] @ half_tau
        self._lambda_grad[1] += np.triu(aux) + np.tril(aux, k=-1).T
        self._lambda_grad[1] *= self.norm_factor
        self._lambda_grad[1][self.tril_inds] = self._lambda_grad[1].T[self.tril_inds]

    def _expected_natural_parameter(self, lambda_, nu):
        return ((nu + 1) * lambda_[0]) / (lambda_[1] * nu - lambda_[0] ** 2), \
               -(nu * (nu + 1)) / (2 * (lambda_[1] * nu - lambda_[0] ** 2))

    def _expected_normalizing_constant(self, lambda_, nu):
        x = lambda_[0]
        y = lambda_[1]
        z = nu
        return - ((x ** 2) * (np.log((y * z - x ** 2) / (2 * z)) - vectorized_digamma((z + 1) / 2) - 1) +
                  y * (z * (vectorized_digamma((z + 1) / 2) - np.log((y * z - x ** 2) / (2 * z))) - 1)) / \
               (2 * (x ** 2 - y * z))

        #return - (1 / 2) * ((lambda_[1] + lambda_[0] ** 2) / (lambda_[1] * nu - lambda_[0] ** 2) -
        #                    vectorized_digamma((nu + 1) / 2) -
        #                    np.log((2 * nu) / (lambda_[1] * nu - lambda_[0] ** 2)))

    def _initialize_weighted_model(self):
        self._nu[:] = self._a
        self._lambda[0] = self._tau.T @ self._T[0] @ self._tau
        self._lambda[1] = self._tau.T @ self._T[1] @ self._tau
        np.fill_diagonal(self._lambda[0], self._lambda[0].diagonal() / 2)
        np.fill_diagonal(self._lambda[1], self._lambda[1].diagonal() / 2)
        self._a += 1 / 2
        self._b += 1 / 2
        self._alpha += 1 / 2
        self._lambda[0] += self._lambda01
        self._lambda[1] += self._lambda02
        self._nu += self._nu0

    def _add_hyperparams_to_grads(self):
        self._a_grad += 1 / 2
        self._b_grad += 1 / 2
        self._alpha_grad += 1 / 2
        self._lambda_grad[0] += self._lambda01
        self._lambda_grad[1] += self._lambda02
        self._nu_grad += self._nu0

    def update_state(self, rho):
        self._compute_bernoulli_local_params()
        self._compute_weighted_local_params()
        self._normalize_and_update_local_params()
        half_tau = self._compute_bernoulli_natural_grads()
        self._compute_weighted_natural_grads(half_tau)
        self._add_hyperparams_to_grads()
        self._update_bernoulli_model(rho)
        self._update_weighted_model(rho)
        self._sample()
        self._lambda_grad.fill(0.0)