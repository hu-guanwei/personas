from collections import defaultdict
from itertools import combinations
from numba import jit, njit
import numpy as np

def prob_hit(s, r, b):
    return 1 - (1 - s ** r) ** b

class localSensitiveHash(object):
    
    def __init__(self, X):
        self._X = X
        self._n_rows, self._n_cols = X.shape
    
    @jit
    def random_proejction(self, r, b):
        v = np.random.uniform(-1, 1, (self._n_cols, r * b))
        return np.sign((self._X @ v).T)
    
    @jit
    def bucket_bands(self, signature_matrix, P, r, b):
        res = []
        for i in range(b):
            r_a = np.random.randint(low=1, high=P, size=(1, r))
            r_b = np.random.randint(low=1, high=P, size=(r, 1))
            res.append(((r_a @ (signature_matrix[i * r: (i + 1) * r, :] + r_b)) % P).ravel())
        return np.array(res).astype(int)


    @jit
    def inverted_index(self, bucket_bands):
        inverted_index = defaultdict(set)
        for key in range(bucket_bands.shape[1]):
            for value in set(bucket_bands[:, key]):
                inverted_index[value].add(key)
        return inverted_index

    @jit
    def find_pairs(self, inverted_index):
        pairs = set()
        for key in inverted_index.keys():
            users = sorted(inverted_index[key])
            pairs.update(tuple(combinations(users, 2)))
        return pairs
