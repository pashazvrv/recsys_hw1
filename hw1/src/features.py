from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _one_hot(codes: np.ndarray, n_values: int | None = None) -> sp.csr_matrix:
    codes = np.asarray(codes)
    n_values = int(codes.max()) + 1 if n_values is None else n_values
    return sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
                         shape=(len(codes), n_values))


def _qbins(x: pd.Series, n_bins: int) -> np.ndarray:
    """Квантильные бакеты (одинаковые значения попадают в один бакет)."""
    return pd.qcut(x.rank(method="first"), q=n_bins, labels=False).to_numpy()


def item_features(split, catalogue: pd.DataFrame, min_attr_items: int = 30,
                  use_id: bool = True, recent_days: int = 3,
                  extra_item_ids: np.ndarray | None = None) -> tuple[sp.csr_matrix, list[str]]:
    """Признаки объектов: [ID] + тип + доступность + длительность + feature_2..5 +
    частые атрибуты + популярность в истории (за всё время и за последние recent_days дней).

    extra_item_ids — объекты вне истории (холодные): их строки добавляются в конец, one-hot ID
    у них нулевой, популярность = 0; бакеты считаются по общему множеству объектов."""
    all_ids = split.item_ids if extra_item_ids is None else np.concatenate([split.item_ids, extra_item_ids])
    n_hist = split.n_items
    cat = catalogue.loc[all_ids]
    blocks, names = [], []

    def add(m, prefix):
        blocks.append(sp.csr_matrix(m, dtype=np.float32))
        names.extend(f"{prefix}_{j}" for j in range(m.shape[1]))

    if use_id:
        add(sp.vstack([sp.identity(n_hist, format="csr"),
                       sp.csr_matrix((len(cat) - n_hist, n_hist))]).tocsr(), "item_id")
    add(_one_hot(pd.Categorical(cat["type"], categories=["movie", "series", "multipart_movie"]).codes, 3),
        "type")
    avail = np.stack([cat["availability"].apply(lambda a, v=v: v in a).to_numpy()
                      for v in ["purchase", "rent", "subscription"]], axis=1).astype(np.float32)
    add(avail, "avail")
    add(_one_hot(_qbins(cat["duration"], 5), 5), "duration_q")
    for f in ["feature_2", "feature_3", "feature_4"]:
        add(_one_hot(_qbins(cat[f], 5), 5), f"{f}_q")
    add(_one_hot(pd.factorize(cat["feature_5"].round(3), sort=True)[0]), "feature_5")
    # атрибуты (анонимные id жанров/людей и т.п.): только достаточно частые, вес 1/sqrt(n)
    attrs = cat["attributes"].explode()
    freq = attrs.value_counts()
    keep = freq.index[freq >= min_attr_items]
    pos = {a: j for j, a in enumerate(keep)}
    rows, cols = [], []
    for r, lst in enumerate(cat["attributes"]):
        for a in lst:
            if a in pos:
                rows.append(r); cols.append(pos[a])
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(cat), len(keep)))
    n = np.asarray(A.sum(1)).ravel()
    A = sp.diags(1 / np.sqrt(np.maximum(n, 1))) @ A
    add(A, "attr")
    pop = np.zeros(len(cat))
    pop[:n_hist] = np.asarray((split.X > 0).sum(0)).ravel()
    add(_one_hot(_qbins(pd.Series(np.log1p(pop)), 10), 10), "pop_q")
    # «тренд»: популярность за последние recent_days дней истории — каталог за месяц
    # заметно обновляется, и недавно популярное важнее популярного месяц назад
    h = split.history
    recent = h[h["ts"] >= split.cutoff_ts - recent_days * 86_400]
    pop_recent = np.bincount(split.item_index(recent["item_id"].to_numpy()), minlength=n_hist)
    pop_recent = np.concatenate([pop_recent, np.zeros(len(cat) - n_hist)])
    add(_one_hot(_qbins(pd.Series(np.log1p(pop_recent)), 10), 10), "pop_recent_q")
    return sp.hstack(blocks).tocsr().astype(np.float32), names


def user_features(split, use_id: bool = True) -> tuple[sp.csr_matrix, list[str]]:
    """Признаки пользователей по истории: [ID] + активность + доли режимов потребления
    (подписка / покупка / аренда) + основной тип устройства + доля сериалов."""
    h = split.history
    u = split.user_index(h["user_id"].to_numpy())
    n_users = split.n_users
    blocks, names = [], []

    def add(m, prefix):
        blocks.append(sp.csr_matrix(m, dtype=np.float32))
        names.extend(f"{prefix}_{j}" for j in range(m.shape[1]))

    if use_id:
        add(sp.identity(n_users, format="csr"), "user_id")
    cnt = np.bincount(u, minlength=n_users)
    add(_one_hot(_qbins(pd.Series(np.log1p(cnt)), 5), 5), "activity_q")
    modes = np.zeros((n_users, 3), dtype=np.float32)
    for j, m in enumerate(["S", "P", "R"]):
        modes[:, j] = np.bincount(u, weights=(h["consumption_mode"] == m).to_numpy(), minlength=n_users)
    add(modes / np.maximum(cnt, 1)[:, None], "mode_share")
    dev = (pd.DataFrame({"u": u, "d": h["device_type"].to_numpy()})
           .groupby("u")["d"].agg(lambda s: s.value_counts().index[0]))
    dev_codes = np.zeros(n_users, dtype=int)
    dev_codes[dev.index.to_numpy()] = dev.to_numpy()
    add(_one_hot(dev_codes, 7), "device")
    series_share = np.bincount(u, weights=(h["item_type"] != "movie").to_numpy(), minlength=n_users)
    add((series_share / np.maximum(cnt, 1))[:, None], "series_share")
    return sp.hstack(blocks).tocsr().astype(np.float32), names
