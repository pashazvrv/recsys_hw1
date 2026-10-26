from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import is_relevant
from .metrics import all_metrics


@dataclass
class Split:
    history: pd.DataFrame            # взаимодействия, на которых учатся модели
    target: pd.DataFrame             # целевой период (validation или test)
    user_ids: np.ndarray             # внешние id пользователей истории (индекс = позиция)
    item_ids: np.ndarray             # внешние id объектов истории (индекс = позиция)
    X: sp.csr_matrix                 # бинарная матрица user x item по истории
    eval_users: np.ndarray           # индексы оцениваемых пользователей
    relevant: list                   # список множеств индексов релевантных объектов
    cutoff_ts: float                 # момент рекомендации (начало целевого периода)
    stats: dict = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def user_index(self, ids) -> np.ndarray:
        return np.searchsorted(self.user_ids, ids)

    def item_index(self, ids) -> np.ndarray:
        return np.searchsorted(self.item_ids, ids)

    def matrix(self, values=None, df: pd.DataFrame | None = None) -> sp.csr_matrix:
        """Матрица user x item по df (по умолчанию — история) с заданными значениями."""
        df = self.history if df is None else df
        vals = np.ones(len(df), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
        return sp.csr_matrix((vals, (self.user_index(df["user_id"].to_numpy()),
                                     self.item_index(df["item_id"].to_numpy()))),
                             shape=(self.n_users, self.n_items))


def make_split(history: pd.DataFrame, target: pd.DataFrame) -> Split:
    user_ids = np.sort(history["user_id"].unique())
    item_ids = np.sort(history["item_id"].unique())

    rel = target[is_relevant(target)]
    warm = rel["user_id"].isin(user_ids) & rel["item_id"].isin(item_ids)
    rel_warm = rel[warm]
    seen = pd.MultiIndex.from_frame(history[["user_id", "item_id"]])
    rel_warm = rel_warm[~pd.MultiIndex.from_frame(rel_warm[["user_id", "item_id"]]).isin(seen)]

    u_idx = np.searchsorted(user_ids, rel_warm["user_id"].to_numpy())
    i_idx = np.searchsorted(item_ids, rel_warm["item_id"].to_numpy())
    groups = pd.Series(i_idx).groupby(u_idx).apply(set)
    eval_users = groups.index.to_numpy()
    relevant = groups.tolist()

    X = sp.csr_matrix((np.ones(len(history), dtype=np.float32),
                       (np.searchsorted(user_ids, history["user_id"].to_numpy()),
                        np.searchsorted(item_ids, history["item_id"].to_numpy()))),
                      shape=(len(user_ids), len(item_ids)))
    stats = {
        "history_interactions": len(history),
        "target_interactions": len(target),
        "target_relevant": len(rel),
        "relevant_cold_user_or_item_share": 1 - warm.mean() if len(rel) else 0.0,
        "eval_users": len(eval_users),
        "eval_relevant_pairs": int(sum(len(s) for s in relevant)),
    }
    return Split(history, target, user_ids, item_ids, X, eval_users, relevant,
                 float(target["ts"].min()), stats)


def recommend_all(model, split: Split, k: int = 10, batch_size: int = 4096) -> np.ndarray:
    recs = []
    for start in range(0, len(split.eval_users), batch_size):
        users = split.eval_users[start:start + batch_size]
        recs.append(model.recommend(users, k=k, exclude=split.X))
    return np.vstack(recs)


def evaluate(model, split: Split, k: int = 10) -> dict:
    t = time.time()
    recs = recommend_all(model, split, k)
    res = all_metrics(recs, split.relevant, k)
    res["predict_sec"] = round(time.time() - t, 1)
    return res
