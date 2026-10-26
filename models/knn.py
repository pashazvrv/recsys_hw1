import numpy as np
import scipy.sparse as sp

from .base import BaseRecommender


class ItemKNN(BaseRecommender):
    def __init__(self, k_neighbors: int = 100, similarity: str = "cosine", shrink: float = 0.0):
        self.k_neighbors = k_neighbors
        self.similarity = similarity
        self.shrink = shrink

    def fit(self, X: sp.csr_matrix, **kwargs):
        Xb = (X > 0).astype(np.float32).tocsc()
        co = (Xb.T @ Xb).toarray()                      
        cnt = np.diag(co).copy()                         
        if self.similarity == "cosine":
            norm = np.sqrt(cnt)
            denom = np.outer(norm, norm) + self.shrink
        elif self.similarity == "jaccard":
            denom = cnt[:, None] + cnt[None, :] - co + self.shrink
        else:
            raise ValueError(self.similarity)
        sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
        np.fill_diagonal(sim, 0.0)

        k = min(self.k_neighbors, sim.shape[0] - 1)
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(sim.shape[0]), k)
        vals = np.take_along_axis(sim, idx, axis=1).ravel()
        self.S_ = sp.csr_matrix((vals, (rows, idx.ravel())), shape=sim.shape)
        self.S_.eliminate_zeros()
        self.X_ = Xb.tocsr()
        return self

    def score(self, user_ids):
        return (self.X_[user_ids] @ self.S_).toarray()
