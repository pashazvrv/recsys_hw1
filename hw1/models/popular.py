import numpy as np

from .base import BaseRecommender


class Popular(BaseRecommender):
    def fit(self, X, **kwargs):
        self.pop_ = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, user_ids):
        return np.tile(self.pop_, (len(user_ids), 1))
