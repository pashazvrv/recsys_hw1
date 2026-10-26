from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .base import BaseRecommender


class _Buckets:

    def __init__(self, Cm1: sp.csr_matrix, CP: sp.csr_matrix, max_cells: int = 1_000_000):
        Cm1, CP = Cm1.tocsr(), CP.tocsr()
        assert np.array_equal(Cm1.indptr, CP.indptr) and np.array_equal(Cm1.indices, CP.indices)
        lens = np.diff(Cm1.indptr)
        self.n = Cm1.shape[0]
        self.empty = torch.from_numpy(np.where(lens == 0)[0])
        self.groups = []
        order = np.argsort(lens, kind="stable")
        order = order[lens[order] > 0]
        bounds = 2 ** np.arange(0, 40)
        b_of = np.searchsorted(bounds, lens[order])         
        for b in np.unique(b_of):
            rows = order[b_of == b]
            L = int(lens[rows].max())
            step = max(1, max_cells // L)
            for s in range(0, len(rows), step):
                r = rows[s:s + step]
                idx = np.zeros((len(r), L), dtype=np.int64)
                w = np.zeros((len(r), L), dtype=np.float32)
                cp = np.zeros((len(r), L), dtype=np.float32)
                for t, row in enumerate(r):
                    a, e = Cm1.indptr[row], Cm1.indptr[row + 1]
                    idx[t, : e - a] = Cm1.indices[a:e]
                    w[t, : e - a] = Cm1.data[a:e]
                    cp[t, : e - a] = CP.data[a:e]
                self.groups.append((torch.from_numpy(r), torch.from_numpy(idx),
                                    torch.from_numpy(w), torch.from_numpy(cp)))


@torch.no_grad()
def _solve_side(F: torch.Tensor, buckets: _Buckets, reg: float) -> torch.Tensor:
    """x = (FᵀF + Σ (c−1) f fᵀ + λI)^{-1} Σ c p f  для всех строк сразу (по бакетам)."""
    k = F.shape[1]
    base = F.T @ F + reg * torch.eye(k, dtype=F.dtype)
    out = torch.zeros((buckets.n, k), dtype=F.dtype)
    for rows, idx, w, cp in buckets.groups:
        Fg = F[idx]                                               # (B, L, k)
        A = torch.bmm((Fg * w[..., None]).transpose(1, 2), Fg) + base
        b = (Fg * cp[..., None]).sum(1)
        out[rows] = torch.linalg.solve(A, b)
    return out


class ImplicitALS(BaseRecommender):
    def __init__(self, n_factors: int = 64, n_iters: int = 15, reg: float = 0.1,
                 alpha: float = 10.0, random_state: int = 42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.alpha = alpha
        self.random_state = random_state

    def _matrices(self, X: sp.csr_matrix, P: sp.csr_matrix | None, C: sp.csr_matrix | None):
        X = X.tocsr().astype(np.float32)
        X.sort_indices()
        r = X.copy()
        r.data = (r.data > 0).astype(np.float32) if C is None else C.tocsr()[X.nonzero()].A1.astype(np.float32)
        conf = r.copy()
        conf.data = 1.0 + self.alpha * r.data                          # c_ui = 1 + α r_ui
        pref = X.copy()
        pref.data = np.ones_like(X.data) if P is None else np.asarray(P.tocsr()[X.nonzero()]).ravel().astype(np.float32)
        Cm1 = conf.copy()
        Cm1.data = conf.data - 1.0
        CP = conf.copy()
        CP.data = conf.data * pref.data
        return Cm1, CP

    def fit(self, X: sp.csr_matrix, P: sp.csr_matrix | None = None,
            C: sp.csr_matrix | None = None, **kwargs):
        Cm1, CP = self._matrices(X, P, C)
        Cm1_T, CP_T = Cm1.T.tocsr(), CP.T.tocsr()
        Cm1_T.sort_indices(); CP_T.sort_indices()
        users_b, items_b = _Buckets(Cm1, CP), _Buckets(Cm1_T, CP_T)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        n_users, n_items = X.shape
        k = self.n_factors
        U = torch.from_numpy((rng.standard_normal((n_users, k)) * 0.01).astype(np.float32))
        V = torch.from_numpy((rng.standard_normal((n_items, k)) * 0.01).astype(np.float32))
        for _ in range(self.n_iters):
            U = _solve_side(V, users_b, self.reg)
            V = _solve_side(U, items_b, self.reg)
        self.user_factors_, self.item_factors_ = U.numpy(), V.numpy()
        return self

    def score(self, user_ids):
        return self.user_factors_[user_ids] @ self.item_factors_.T

    def loss(self, X, P=None, C=None) -> float:
        """Значение целевой функции (для проверки сходимости на небольших данных)."""
        Cm1, CP = self._matrices(X, P, C)
        U, V = self.user_factors_, self.item_factors_
        full = (U @ V.T) ** 2                              # вклад c=1, p=0 по всем парам
        rows, cols = Cm1.nonzero()
        pred = np.einsum("ij,ij->i", U[rows], V[cols])
        c = Cm1.data + 1.0
        p = CP.data / c
        obs = c * (p - pred) ** 2 - pred ** 2              # заменяем вклад наблюдаемых пар
        return float(full.sum() + obs.sum() + self.reg * ((U ** 2).sum() + (V ** 2).sum()))
