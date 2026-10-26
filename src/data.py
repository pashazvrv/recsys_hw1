from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

TRANSACTION_DTYPES = {
    "element_uid": "int32",
    "user_uid": "int32",
    "consumption_mode": "category",
    "ts": "float64",
    "watched_time": "int64",
    "device_type": "int16",
    "device_manufacturer": "int16",
}


def load_transactions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "transactions.csv", dtype=TRANSACTION_DTYPES)


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "ratings.csv",
                       dtype={"user_uid": "int32", "element_uid": "int32", "rating": "int8", "ts": "float64"})


def load_bookmarks(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "bookmarks.csv",
                       dtype={"user_uid": "int32", "element_uid": "int32", "ts": "float64"})


def load_catalogue(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    with open(Path(data_dir) / "catalogue.json", encoding="utf-8") as f:
        raw = json.load(f)
    cat = pd.DataFrame.from_dict(raw, orient="index")
    cat.index = cat.index.astype("int32")
    cat.index.name = "element_uid"
    return cat.sort_index()


def load_raw(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """читает 4 файла и возвращаем таблицы оттуда"""
    return {
        "transactions": load_transactions(data_dir),
        "ratings": load_ratings(data_dir),
        "bookmarks": load_bookmarks(data_dir),
        "catalogue": load_catalogue(data_dir),
    }


SECONDS_IN_DAY = 86_400


def prepare_interactions(transactions: pd.DataFrame, ratings: pd.DataFrame,
                         bookmarks: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на пару (user, item) из transactions + сигналы из ratings/bookmarks.

    Добавляет:
      day          — номер дня от начала лога ;
      duration_sec — длительность объекта;
                     объекты с нулевой длительностью (в каталоге их 114, ~1% просмотров) удаляются:
                     для них нельзя посчитать долю просмотра;
      watch_ratio  — watched_time / duration_sec (для сериалов — «сколько серий посмотрено»);
      rating       — оценка 0–10 пары (NaN, если нет), rating_ts;
      bookmarked   — была ли закладка, bookmark_ts.
    """
    df = transactions.rename(columns={"user_uid": "user_id", "element_uid": "item_id"}).copy()
    t0 = df["ts"].min()
    df["day"] = ((df["ts"] - t0) // SECONDS_IN_DAY).astype("int16")

    cat = catalogue[["type", "duration"]].copy()
    df = df.join(cat.rename(columns={"type": "item_type"}), on="item_id")
    df = df[df["duration"] > 0].reset_index(drop=True)
    df["duration_sec"] = (df.pop("duration") * 60).astype("float32")
    df["watch_ratio"] = (df["watched_time"] / df["duration_sec"]).astype("float32")

    r = ratings.rename(columns={"user_uid": "user_id", "element_uid": "item_id", "ts": "rating_ts"})
    b = bookmarks.rename(columns={"user_uid": "user_id", "element_uid": "item_id", "ts": "bookmark_ts"})
    df = df.merge(r, on=["user_id", "item_id"], how="left")
    df = df.merge(b, on=["user_id", "item_id"], how="left")
    df["bookmarked"] = df["bookmark_ts"].notna()
    df.attrs["t0"] = t0
    return df


def is_completed(df: pd.DataFrame, movie_threshold: float = 0.5,
                 episodic_threshold: float = 1.0) -> pd.Series:
    """Досмотр: фильм — не меньше movie_threshold длительности;
    сериал / многосерийный фильм (длительность = одна серия) — не меньше episodic_threshold серий."""
    thr = np.where(df["item_type"] == "movie", movie_threshold, episodic_threshold)
    return df["watch_ratio"] >= thr


def is_relevant(df: pd.DataFrame, like_rating: int = 8, dislike_rating: int = 4) -> pd.Series:
    """объект досмотрен (is_completed) ИЛИ оценён на >= like_rating,
    и при этом оценен на > dislike_rating."""
    liked = df["rating"] >= like_rating
    disliked = df["rating"] <= dislike_rating
    return (is_completed(df) | liked) & ~disliked

def make_subsample(df: pd.DataFrame, n_users: int, seed: int, filter_before_day: int,
                   min_user_items: int = 5, min_item_users: int = 10) -> pd.DataFrame:
    """Воспроизводимая подвыборка. Фильтры считаются по периоду до filter_before_day
    (начало validation), чтобы отбор не использовал информацию из будущего:
    1) пользователи с >= min_user_items объектов до filter_before_day;
    2) случайно (seed) выбираем n_users из них;
    3) объекты, которые до filter_before_day смотрели >= min_item_users выбранных пользователей;
    4) берём все взаимодействия (за весь период) выбранных пользователей с выбранными объектами."""
    early = df[df["day"] < filter_before_day]
    user_cnt = early.groupby("user_id")["item_id"].size()
    eligible = np.sort(user_cnt.index[user_cnt >= min_user_items].to_numpy())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    early_sub = early[early["user_id"].isin(chosen)]
    item_cnt = early_sub.groupby("item_id")["user_id"].size()
    keep_items = item_cnt.index[item_cnt >= min_item_users]
    sub = df[df["user_id"].isin(chosen) & df["item_id"].isin(keep_items)]
    return sub.reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_end_day: int, val_end_day: int):
    """Глобальное временное разбиение по дням:
    train — day < train_end_day; validation — train_end_day <= day < val_end_day; test — day >= val_end_day."""
    train = df[df["day"] < train_end_day]
    val = df[(df["day"] >= train_end_day) & (df["day"] < val_end_day)]
    test = df[df["day"] >= val_end_day]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
