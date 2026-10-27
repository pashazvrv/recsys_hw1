from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .base import BaseRecommender


def _to_padded(F: sp.csr_matrix):
    
    F = F.tocsr()
    lens = np.diff(F.indptr)
    L = max(int(lens.max()), 1)
    idx = np.zeros((F.shape[0], L), dtype=np.int64)
    w = np.zeros((F.shape[0], L), dtype=np.float32)
    for r in range(F.shape[0]):
        s, e = F.indptr[r], F.indptr[r + 1]
        idx[r, : e - s] = F.indices[s:e]
        w[r, : e - s] = F.data[s:e]
    return torch.from_numpy(idx), torch.from_numpy(w)


class FactorizationMachine(BaseRecommender):
    def __init__(self, n_factors: int = 64, n_epochs: int = 10, lr: float = 5e-3,
                 reg: float = 1e-5, batch_size: int = 4096, random_state: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.random_state = random_state

    def _embed(self, idx, w):
        V = self.V[idx] * w[..., None]                    
        e = V.sum(1)                                      #  sum v_f x_f
        sq = (V ** 2).sum((1, 2))                         # sum ||v_f x_f||²
        lin = (self.w[idx] * w).sum(1)                    #  sum w_f x_f
        return e, lin, sq

    def _all_items(self):
        Vi = self.V[self.n_uf_:]
        wi = self.w[self.n_uf_:]
        e = torch.sparse.mm(self.IF_, Vi)          
        sq = torch.sparse.mm(self.IF2_, (Vi ** 2).sum(1, keepdim=True)).squeeze(1)
        lin = torch.sparse.mm(self.IF_, wi[:, None]).squeeze(1)
        b = lin + 0.5 * ((e ** 2).sum(1) - sq)
        return e, b

    def fit(self, X: sp.csr_matrix, user_features: sp.csr_matrix = None,
            item_features: sp.csr_matrix = None, verbose: bool = False, **kwargs):

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        n_users, n_items = X.shape
        if user_features is None:
            user_features = sp.identity(n_users, format="csr", dtype=np.float32)
        if item_features is None:
            item_features = sp.identity(n_items, format="csr", dtype=np.float32)
        n_uf = user_features.shape[1]
        n_feat = n_uf + item_features.shape[1]
        self.user_idx, self.user_w = _to_padded(user_features)
        self.n_uf_ = n_uf
        itf = item_features.tocoo()
        ind = torch.from_numpy(np.vstack([itf.row, itf.col]).astype(np.int64))
        self.IF_ = torch.sparse_coo_tensor(ind, torch.from_numpy(itf.data.astype(np.float32)),
                                           itf.shape).coalesce()
        self.IF2_ = torch.sparse_coo_tensor(ind, torch.from_numpy((itf.data ** 2).astype(np.float32)),
                                            itf.shape).coalesce()

        k = self.n_factors
        self.V = torch.nn.Parameter(torch.randn(n_feat, k) * 0.01)
        self.w = torch.nn.Parameter(torch.zeros(n_feat))
        opt = torch.optim.Adam([self.V, self.w], lr=self.lr)

        Xc = X.tocoo()
        pos_u, pos_i = Xc.row.astype(np.int64), Xc.col.astype(np.int64)
        n = len(pos_u)
        self.history_ = []
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            neg = rng.integers(0, n_items, size=n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                u = torch.from_numpy(pos_u[b])
                i = torch.from_numpy(pos_i[b])
                j = torch.from_numpy(neg[b])
                e_u, _, _ = self._embed(self.user_idx[u], self.user_w[u])
                e_all, b_all = self._all_items()
                e_i, b_i, e_j, b_j = e_all[i], b_all[i], e_all[j], b_all[j]
                diff = b_i - b_j + (e_u * (e_i - e_j)).sum(1)
                loss = -torch.nn.functional.logsigmoid(diff).mean()
                reg = self.reg * ((e_u ** 2).sum(1) + (e_i ** 2).sum(1) + (e_j ** 2).sum(1)).mean()
                opt.zero_grad()
                (loss + reg).backward()
                opt.step()
                total += loss.item() * len(b)
            self.history_.append(total / n)
            if verbose:
                print(f"epoch {epoch + 1}: BPR loss {self.history_[-1]:.4f}")
        self._precompute()
        return self

    @torch.no_grad()
    def _precompute(self):
        e_i, b_i = self._all_items()
        e_u, _, _ = self._embed(self.user_idx, self.user_w)
        self.item_emb_, self.item_bias_ = e_i.numpy(), b_i.numpy()
        self.user_emb_ = e_u.numpy()

    @torch.no_grad()
    def item_representations(self, item_features: sp.csr_matrix):

        F = item_features.tocsr().astype(np.float32)
        Vi = self.V[self.n_uf_:].detach().numpy()
        wi = self.w[self.n_uf_:].detach().numpy()
        e = np.asarray(F @ Vi)
        sq = np.asarray(F.multiply(F) @ (Vi ** 2).sum(1)).ravel()
        b = np.asarray(F @ wi).ravel() + 0.5 * ((e ** 2).sum(1) - sq)
        return e, b

    def score(self, user_ids):
        return self.user_emb_[user_ids] @ self.item_emb_.T + self.item_bias_[None, :]
