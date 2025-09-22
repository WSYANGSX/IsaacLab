import numpy as np
from sklearn.mixture import GaussianMixture

X = np.array([[1, 3], [1, 5], [1, 0.5], [10, 6], [10, 7], [10, 3]])
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
print(gm.means_)