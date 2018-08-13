
""" Collection of random number generation methodologies.
"""

#%% imports
import numpy as np

# make output stable across runs
np.random.seed(42)

# to plot pretty figures
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


#%% exponential continuous random variable

# use exponential distribution when you have some idea what the scale of the hyperparameter should be
# log of distribution shows most values aroung exp(-2) to exp(+2), i.e. ~ 0.1 to 7.4

from scipy.stats import expon, reciprocal

expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)

plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)

plt.show()


#%% reciprocal continuous random variable

# use reciprocal distribution when you have no idea what the scale of the hyperparameter should be
# log of the samples roughly constant as scale of the samples picked from a uniform distribution

reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)

plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)

plt.show()

#%% linear looking data

import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100,  1)


# nonlinear and noisy dataset generated from a quadratic equation

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

