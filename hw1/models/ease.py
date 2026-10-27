import numpy as np
import scipy.sparse as sp

from .base import BaseRecommender


class EASE(BaseRecommender):
    def __init__(self, reg: float = 500.0):
        self.reg = reg

    def fit(self, X: sp.csr_matrix, **kwargs):
        Xb = (X > 0).astype(np.float64).tocsr()
        G = (Xb.T @ Xb).toarray()
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        self.B_ = B.astype(np.float32)
        self.X_ = Xb.astype(np.float32)
        return self

    def score(self, user_ids):
        return np.asarray(self.X_[user_ids] @ self.B_)
