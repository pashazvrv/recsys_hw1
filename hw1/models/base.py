import numpy as np
import scipy.sparse as sp


class BaseRecommender:

    def fit(self, X: sp.csr_matrix, **kwargs):
        raise NotImplementedError

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_ids: np.ndarray, k: int = 10,
                  exclude: sp.csr_matrix | None = None) -> np.ndarray:
        scores = self.score(user_ids).astype(np.float32, copy=True)
        if exclude is not None:
            rows = exclude[user_ids]
            r, c = rows.nonzero()
            scores[r, c] = -np.inf
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        return np.take_along_axis(top, order, axis=1)
