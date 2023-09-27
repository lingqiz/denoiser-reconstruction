import numpy as np
from scipy.stats import gennorm

# Define bivariate distributions

'''
2D rotation matrix
'''
def rotation(theta):
    theta = np.radians(theta)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

'''
Return samples from a bivariate Normal distribution
'''
def normal(mu, sigma, rho, n):
    x = np.random.normal(mu[0], sigma[0], n)
    y = np.random.normal(mu[1], sigma[1], n)
    z = rho * x + np.sqrt(1 - rho**2) * y
    return np.array([x, z]).T

'''
Sum of two Gaussian ridges
'''
def ridge(range=[-1, 1], theta=0, n=1, balance=False):
    shrink = 0.75
    if balance:
        x = np.random.normal(0, 0.01, n)
        y = np.linspace(range[0], range[1], n)
        sample_1 = np.array([x, y * shrink]).T
        sample_2 = np.array([y, x]).T

    else:
        x = np.random.normal(0, 0.01, n)
        y = np.random.uniform(range[0], range[1], n) * shrink
        sample_1 = np.array([x, y]).T

        x = np.random.uniform(range[0], range[1], n)
        y = np.random.normal(0, 0.01, n)
        sample_2 = np.array([x, y]).T

    sample = np.concatenate((sample_1, sample_2))
    np.random.shuffle(sample)

    # rotate the samples
    rot_mtx = rotation(theta)
    return (rot_mtx @ sample.T).T

'''
Independent sparse marginals
'''
def sparse(shape=1, scale=1, theta=0, n=1):
    # generlized normal density
    x = gennorm.rvs(shape, size=n)
    y = gennorm.rvs(shape, size=n)
    sample = np.array([x, y]).T * scale

    return (rotation(theta) @ sample.T).T

'''
K-sparse density
Compressed sensing
Weiss et al., 2008
'''
def k_sparse(n=1, length=0.5):
    TOTAL = 2 + length * 2
    P_HORI = 2 / TOTAL

    sample = []
    for _ in range(n):
        if np.random.uniform() < P_HORI:
            x = np.random.uniform(-1, 1)
            y = np.random.normal(0, 0.01)

        else:
            x = np.random.normal(0, 0.01)
            y = np.random.uniform(1 - length, 1.0)
            if np.random.uniform() >= 0.5:
                y = -y

        sample.append([x, y])

    return np.array(sample)