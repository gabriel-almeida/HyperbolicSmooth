import numpy as np
import scipy.optimize


def hyperbolic_smooth(fn, h, n_restrictions, x0, lamb1, tau1, r, q,
                      max_iter=100, stop=1e-8, debug=True):
    if not lamb1 > 0:
        raise Exception("Lambda must be positive")
    if not tau1 > 0:
        raise Exception("Tau must be positive")
    if not r > 1:
        raise Exception("r must be greater than 1")
    if not 0 < q < 1:
        raise Exception("q must be between 0 and 1")

    lamb = np.ones(n_restrictions)*lamb1
    tau = np.ones(n_restrictions)*tau1

    last_x = x0
    last_y = 0

    for k in range(max_iter):
        modified_fn = lambda x: fn(x) + np.sum(_hyperbolic_penalty(h(x), lamb, tau))
        optimization_obj = scipy.optimize.minimize(modified_fn, last_x, method='BFGS')

        x = optimization_obj.x
        penalized_y = optimization_obj.fun
        y = fn(x)
        hx = h(x)
        x_step = np.linalg.norm(x - last_x) / (1 + np.linalg.norm(x))
        y_step = np.abs(y - last_y) / (1 + y)

        if debug:
            print()
            print("======", k, "======")
            print("x=", x)
            print("y=", y)
            print("Penalized y=", penalized_y)
            print("H(x)=", hx)
            print("lambda=", lamb)
            print("tau=", tau)
            print("Penalty=", _hyperbolic_penalty(hx, lamb, tau))
            print("Step size in x=", x_step)
            print("Step size in y=", y_step)
            print("fn evaluations=", optimization_obj.nfev)
            print("Solver iterations=", optimization_obj.nit)

        # Preparation to the next iteration
        last_x = x
        last_y = y

        # Stop conditions
        if k > 0 and x_step < stop or y_step < stop:
            break

        # Smoothing update
        for i in range(n_restrictions):
            unfeasible = hx[i] < 0.0
            if unfeasible:
                lamb[i] *= r
            else:
                tau[i] *= q

    return last_x


def _hyperbolic_penalty(y, lamb, tau):
    return -lamb*y + np.sqrt(lamb**2 * y**2 + tau)
