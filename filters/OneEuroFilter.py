import numpy as np

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.2, beta=0.01):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None
        self.dx_prev = 0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def update(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) * self.freq
        cutoff = self.min_cutoff + self.beta * abs(dx)

        alpha = self._alpha(cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        return x_hat
