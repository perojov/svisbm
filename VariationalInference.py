import time
import numpy as np
import matplotlib.pyplot as plt


class VariationalInference:
    def __init__(self, state, simulation_time):
        """
        :param state: Model parameters and update functions.
        :param simulation_time: Timesteps.
        """
        self._simulation_time = simulation_time
        self._state = state

    def run(self):
        perplexity = np.zeros(self._simulation_time)
        perplexity[0] = self._state.compute_perplexity()
        for t in range(1, self._simulation_time):
            start = time.time()
            self._state.update()
            perplexity[t] = self._state.compute_perplexity()
            end = time.time()
            print('Iteration %d took %.1f seconds.' % (t, end - start))
        plt.plot(perplexity)
        plt.show()


class StochasticVariationalInference:
    def __init__(self, state, simulation_time=1000, tau0=128, kappa=0.75):
        """
        :param state: Model parameters and update functions.
        :param simulation_time: Timesteps.
        :param tau0: Learning rate parameter.
        :param kappa: Learning rate parameter.
        """
        self._state = state
        self._simulation_time = simulation_time
        self._tau0 = tau0
        self._kappa = kappa

        # Learning rate.
        self.rho = lambda t: (t + self._tau0) ** -self._kappa

    def run(self):
        for t in range(1, self._simulation_time):
            start = time.time()
            self._state.update_state(rho=self.rho(t))
            end = time.time()
            print('Iteration %d took %.1f seconds.' % (t, end - start))

