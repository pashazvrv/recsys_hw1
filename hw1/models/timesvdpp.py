
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .base import BaseRecommender

DAY = 86_400.0


class TimeSVDpp(BaseRecommender):
    def __init__(self, n_factors: int = 32, n_epochs: int = 8, lr: float = 3e-3, reg: float = 1e-4,
                 batch_size: int = 16_384, n_bins: int = 5, beta: float = 0.4,
                 use_time: bool = True, use_implicit: bool = True, random_state: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.n_bins = n_bins
        self.beta = beta
        self.use_time = use_time
        self.use_implicit = use_implicit
        self.random_state = random_state

    def _dev(self, t, tbar):
        d = t - tbar
        return np.sign(d) * np.abs(d) ** self.beta

    def _bins(self, t):
        b = np.floor((t - self.t0_) / max(self.cutoff_ - self.t0_, 1e-9) * self.n_bins).astype(np.int64)
        return np.clip(b, 0, self.n_bins - 1)

    def fit(self, X: sp.csr_matrix, split=None, verbose: bool = False, **kwargs):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = X.tocsr().astype(np.float32)
        n_users, n_items = X.shape
        k, B = self.n_factors, self.n_bins

        h = split.history
        ev = pd.DataFrame({"u": split.user_index(h["user_id"].to_numpy()),
                           "i": split.item_index(h["item_id"].to_numpy()),
                           "t": h["ts"].to_numpy() / DAY})
        ev = ev.groupby(["u", "i"], as_index=False)["t"].max()
        pos_u = ev["u"].to_numpy(np.int64)
        pos_i = ev["i"].to_numpy(np.int64)
        pos_t = ev["t"].to_numpy()
        self.cutoff_ = split.cutoff_ts / DAY
        self.t0_ = float(pos_t.min())
        self.tbar_ = np.bincount(pos_u, weights=pos_t, minlength=n_users) / np.maximum(
            np.bincount(pos_u, minlength=n_users), 1)
        pos_bin = self._bins(pos_t)
        pos_dev = self._dev(pos_t, self.tbar_[pos_u]).astype(np.float32)
        deg = np.asarray((X > 0).sum(1)).ravel().astype(np.float32)
        n = len(pos_u)

        init = lambda *s: torch.nn.Parameter(torch.randn(*s) * 0.01)
        self.P, self.A = init(n_users, k), init(n_users, k)     # p_u и alpha_u
        self.Q, self.Y = init(n_items, k), init(n_items, k)     # q_i и y_j
        self.b = torch.nn.Parameter(torch.zeros(n_items))       # b_i
        self.bb = torch.nn.Parameter(torch.zeros(n_items, B))   # b_{i,bin}
        opt = torch.optim.Adam([self.P, self.A, self.Q, self.Y, self.b, self.bb], lr=self.lr)

        self.history_ = []
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            neg = rng.integers(0, n_items, size=n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                ix = perm[s:s + self.batch_size]
                u_np, i_np, j_np = pos_u[ix], pos_i[ix], neg[ix]
                u, i, j = (torch.from_numpy(a) for a in (u_np, i_np, j_np))


                uvec = self.P[u]
                if self.use_time:
                    uvec = uvec + self.A[u] * torch.from_numpy(pos_dev[ix])[:, None]
                z = None
                if self.use_implicit:
                    uu, inv = np.unique(u_np, return_inverse=True)
                    Xb = X[uu].tocoo()
                    Xt = torch.sparse_coo_tensor(np.vstack([Xb.row, Xb.col]).astype(np.int64),
                                                 Xb.data, Xb.shape)
                    S = torch.sparse.mm(Xt, self.Y)[torch.from_numpy(inv)]          # sum_{j∈N(u)} y_j
                    n_wo = torch.from_numpy(np.maximum(deg[u_np] - 1, 1.0))
                    z = (S - self.Y[i]) / n_wo.sqrt()[:, None]
                    uvec = uvec + z

                bi, bj = self.b[i], self.b[j]
                if self.use_time:
                    bin_ = torch.from_numpy(pos_bin[ix])
                    bi = bi + self.bb[i, bin_]
                    bj = bj + self.bb[j, bin_]
                q_i, q_j = self.Q[i], self.Q[j]
                diff = bi - bj + (uvec * (q_i - q_j)).sum(1)
                loss = -torch.nn.functional.logsigmoid(diff).mean()

                r = (self.P[u] ** 2).sum(1) + (q_i ** 2).sum(1) + (q_j ** 2).sum(1) + bi ** 2 + bj ** 2
                if self.use_time:
                    r = r + (self.A[u] ** 2).sum(1)
                if z is not None:
                    r = r + (z ** 2).sum(1)
                reg = self.reg * r.mean()
                opt.zero_grad()
                (loss + reg).backward()
                opt.step()
                total += loss.item() * len(ix)
            self.history_.append(total / n)
            if verbose:
                print(f"epoch {epoch + 1}: BPR loss {self.history_[-1]:.4f}")
        self._precompute(X, deg)
        return self

    @torch.no_grad()
    def _precompute(self, X, deg):
        uvec = self.P.numpy().copy()
        if self.use_time:
            dev = self._dev(self.cutoff_, self.tbar_).astype(np.float32)
            uvec += self.A.numpy() * dev[:, None]
        if self.use_implicit:
            uvec += (X @ self.Y.numpy()) / np.sqrt(np.maximum(deg, 1.0))[:, None]
        bias = self.b.numpy().copy()
        if self.use_time:
            bias += self.bb.numpy()[:, -1]          
        self.user_vec_, self.item_vec_, self.item_bias_ = uvec, self.Q.numpy().copy(), bias

    def score(self, user_ids):
        return self.user_vec_[user_ids] @ self.item_vec_.T + self.item_bias_[None, :]
