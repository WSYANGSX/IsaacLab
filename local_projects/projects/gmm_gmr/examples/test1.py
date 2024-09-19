import matplotlib.pyplot as plt
import numpy as np
from gmr import MVN


mvn = MVN(mean=[0.0], covariance=[[1.0]])
alpha = 0.6827
X = np.linspace(-3, 3, 101)[:, np.newaxis]
P = mvn.to_probability_density(X)

for x, p in zip(X, P):
    conf = mvn.is_in_confidence_region(x, alpha)
    color = "g" if conf else "r"
    plt.plot([x[0], x[0]], [0, p], color=color)

plt.plot(X.ravel(), P)

plt.xlabel("x")
plt.ylabel("Probability Density $p(x)$")
plt.show()