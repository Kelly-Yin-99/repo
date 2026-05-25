import math

class BayesFactor:
    def __init__(self, n, k):
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
        
        self.n = n
        self.k = k
        # Etz et al. (2018) values for the spike interval
        self.a = 0.495
        self.b = 0.505

    def likelihood(self, theta):
        if not isinstance(theta, (int, float)):
            raise ValueError("theta must be numeric")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be between 0 and 1")
        
        # Binomial likelihood: nCk * theta^k * (1-theta)^(n-k)
        comb = math.comb(self.n, self.k)
        return float(comb * (theta**self.k) * ((1 - theta)**(self.n - self.k)))

    def evidence_slab(self):
        # Slab prior: Uniform(0, 1).
        # Evidence = integral from 0 to 1 of nCk * theta^k * (1-theta)^(n-k) * 1 d(theta)
        # This is nCk * Beta(k+1, n-k+1) = n!/(k!(n-k)!) * (k!(n-k)!)/(n+1)! = 1/(n+1)
        return 1.0 / (self.n + 1)

    def evidence_spike(self):
        # Spike prior: Uniform(a, b).
        # Evidence = integral from a to b of Likelihood(theta) * (1 / (b-a)) d(theta)
        if self.a == self.b:
            return float(self.likelihood(self.a))
        
        # Numerical integration using the trapezoidal rule
        # The likelihood function is a polynomial, so high resolution is sufficient.
        steps = 10000
        h = (self.b - self.a) / steps
        total = 0.0
        for i in range(steps + 1):
            theta = self.a + i * h
            weight = 0.5 if (i == 0 or i == steps) else 1.0
            total += weight * self.likelihood(theta)
        
        # integral * (1 / (b-a))
        return (total * h) / (self.b - self.a)

    def bayes_factor(self):
        # Bayes Factor = Evidence Spike / Evidence Slab
        e_spike = self.evidence_spike()
        e_slab = self.evidence_slab()
        
        if e_slab == 0:
            return float('inf')
        
        return e_spike / e_slab