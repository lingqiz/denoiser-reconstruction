import numpy as np

# Define bivariate distributions

'''
Return samples from a bivariate Normal distribution
'''
def normal(mu, sigma, rho, n):
    x = np.random.normal(mu[0], sigma[0], n)
    y = np.random.normal(mu[1], sigma[1], n)
    z = rho * x + np.sqrt(1 - rho**2) * y
    return np.array([x, z]).T