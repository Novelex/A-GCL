"""Tangent + shrinkage transformer for S17. Lives in a REAL module (not __main__)
because joblib.Memory hashes estimators with standard pickle-by-reference: a class
defined in the launching script's __main__ is unpicklable inside loky workers
('not found as __main__.TangentShrink'), which killed the first s17_tangent2 run
right after its reproduction gates passed."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

IU = np.triu_indices(90, k=1)
I90 = np.eye(90)

def sym_fun(Msym, fun):
    lam, V = np.linalg.eigh(Msym)
    return (V * fun(lam)) @ V.T

def logm_spd(M):  return sym_fun(M, np.log)
def inv_sqrtm(M): return sym_fun(M, lambda l: 1.0 / np.sqrt(l))


class TangentShrink(BaseEstimator, TransformerMixin):
    """Shrink toward identity, C' = (1-alpha)C + alpha*I, then log-Euclidean tangent
    projection logm(M^-1/2 C' M^-1/2) with M = expm(mean(logm(C'_train))).
    fit() sees ONLY the training block sklearn hands it, so the reference mean can
    never include a validation or test subject. Returns strict upper triangle k=1."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def _shrink(self, M):
        return (1.0 - self.alpha) * M + self.alpha * I90

    def fit(self, X, y=None):
        M = np.asarray(X).reshape(-1, 90, 90)
        logs = np.stack([logm_spd(self._shrink(m)) for m in M])
        self.Wh_ = inv_sqrtm(sym_fun(logs.mean(0), np.exp))
        self.n_fit_ = len(M)
        return self

    def transform(self, X):
        M = np.asarray(X).reshape(-1, 90, 90)
        out = np.empty((len(M), 4005))
        for j, m in enumerate(M):
            T = logm_spd(self.Wh_ @ self._shrink(m) @ self.Wh_)
            out[j] = ((T + T.T) / 2.0)[IU]
        return out
