import math
from scipy.integrate import quad


class BayesFactor:

    def __init__(self, n, k, a, b):
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        if not isinstance(k, int):
            raise ValueError("k must be an integer")

        if n < 0:
            raise ValueError("n must be non-negative")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > n:
            raise ValueError("k cannot be greater than n")

        if not (0 <= a < b <= 1):
            raise ValueError("invalid interval")

        self.n = n
        self.k = k
        self.a = a
        self.b = b

    def likelihood(self, theta):
        if not isinstance(theta, (int, float)):
            raise ValueError("theta must be numeric")
        if theta < 0 or theta > 1:
            raise ValueError("theta must be between 0 and 1")

        return math.comb(self.n, self.k) * (theta ** self.k) * (
            (1 - theta) ** (self.n - self.k)
        )

    def evidence_slab(self):
        val, _ = quad(self.likelihood, 0, 1)
        return val

    def evidence_spike(self):
        width = self.b - self.a

        def g(x):
            return self.likelihood(x) / width

        val, _ = quad(g, self.a, self.b)
        return val

    def bayes_factor(self):
        base = self.evidence_slab()
        if base == 0:
            raise RuntimeError("zero evidence")

        return self.evidence_spike() / base