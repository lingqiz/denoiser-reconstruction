import numpy as np

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
def ridge(range=[-1, 1], theta=0, n=1):
    x = np.random.normal(0, 0.01, n)
    y = np.random.uniform(range[0], range[1], n)
    sample_1 = np.array([x, y]).T

    x = np.random.uniform(range[0], range[1], n)
    y = np.random.normal(0, 0.01, n)
    sample_2 = np.array([x, y]).T

    sample = np.concatenate((sample_1, sample_2))
    np.random.shuffle(sample)

    rot_mtx = rotation(theta)
    return (rot_mtx @ sample.T).T