import HyperbolicSmooth
import numpy as np


def fn(x):
    y = 4*(x[0] - 4)**2 + 25*(x[1] + 5)**2
    return y


def g(x):
    h = np.zeros(3)
    h[0] = x[0]
    h[1] = x[1]
    h[2] = -(x[0]**2 + x[1]**2) + 1
    return h


if __name__ == "__main__":
    HyperbolicSmooth.hyperbolic_smooth(fn, g, 3, x0=[0, 0], lamb1=5,
                                       tau1=5, r=10, q=0.1, stop=1e-4, debug=True)
