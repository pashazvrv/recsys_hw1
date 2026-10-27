"""Метрики качества top-K рекомендаций (лекция 2, семинар 2).

recs     — массив (n_users, K) индексов рекомендованных объектов, по убыванию скора;
relevant — список множеств релевантных объектов на test для тех же пользователей.
"""
from __future__ import annotations

import numpy as np


def _hits(recs: np.ndarray, relevant: list[set], k: int) -> np.ndarray:
    """Матрица (n_users, k): 1, если объект на позиции релевантен."""
    return np.array([[item in rel for item in row[:k]] for row, rel in zip(recs, relevant)],
                    dtype=np.float64)


def ndcg_at_k(recs: np.ndarray, relevant: list[set], k: int = 10) -> float:
    """NDCG@K с бинарной релевантностью: DCG = sum rel_i / log2(i + 1), нормировка на идеальный DCG."""
    hits = _hits(recs, relevant, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discounts
    n_rel = np.array([min(len(rel), k) for rel in relevant])
    idcg = np.array([discounts[:n].sum() for n in n_rel])
    return float(np.mean(np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)))


def ndcg_per_user(recs: np.ndarray, relevant: list[set], k: int = 10) -> np.ndarray:
    """NDCG@K для каждого пользователя отдельно (для бутстрап-оценки разниц между моделями)."""
    hits = _hits(recs, relevant, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discounts
    n_rel = np.array([min(len(rel), k) for rel in relevant])
    idcg = np.array([discounts[:n].sum() for n in n_rel])
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Парный бутстрап по пользователям: разница средних mean(a) − mean(b) и 95%-интервал."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[t] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"разница": float(a.mean() - b.mean()), "95% CI низ": float(lo), "95% CI верх": float(hi),
            "доля бутстрапа > 0": float((diffs > 0).mean())}


def recall_at_k(recs: np.ndarray, relevant: list[set], k: int = 10) -> float:
    """Recall@K: доля релевантных объектов пользователя, попавших в top-K.
    Знаменатель min(|rel|, K), чтобы метрика достигала 1 при |rel| > K."""
    hits = _hits(recs, relevant, k).sum(axis=1)
    denom = np.array([min(len(rel), k) for rel in relevant], dtype=np.float64)
    return float(np.mean(np.divide(hits, denom, out=np.zeros_like(hits), where=denom > 0)))


def precision_at_k(recs: np.ndarray, relevant: list[set], k: int = 10) -> float:
    return float(np.mean(_hits(recs, relevant, k).sum(axis=1) / k))


def hit_rate_at_k(recs: np.ndarray, relevant: list[set], k: int = 10) -> float:
    return float(np.mean(_hits(recs, relevant, k).max(axis=1)))


def all_metrics(recs, relevant, k: int = 10) -> dict:
    return {
        f"NDCG@{k}": ndcg_at_k(recs, relevant, k),
        f"Recall@{k}": recall_at_k(recs, relevant, k),
        f"Precision@{k}": precision_at_k(recs, relevant, k),
        f"HitRate@{k}": hit_rate_at_k(recs, relevant, k),
    }
