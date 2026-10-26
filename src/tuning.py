from __future__ import annotations

import json
import time
from pathlib import Path

import optuna
import pandas as pd

from .evaluation import evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(make_model, space, split, n_trials: int, seed: int, fit_kwargs: dict | None = None,
         metric: str = "NDCG@10", k: int = 10, cache: str | Path | None = None,
         enqueue: list[dict] | None = None):

    if cache is not None and Path(cache).exists():
        trials = pd.read_csv(cache)
        best = json.loads(Path(cache).with_suffix(".json").read_text(encoding="utf-8"))
        print(f"[кеш] результаты подбора прочитаны из {cache}")
        return best, trials
    fit_kwargs = fit_kwargs or {}
    rows = []

    def objective(trial):
        params = space(trial)
        t = time.time()
        model = make_model(params).fit(split.X, **fit_kwargs)
        fit_sec = time.time() - t
        res = evaluate(model, split, k)
        rows.append({"trial": trial.number, **params, **res, "fit_sec": round(fit_sec, 1)})
        return res[metric]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    for params in enqueue or []:
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=n_trials)
    trials = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    best = study.best_params
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        trials.to_csv(cache, index=False)
        Path(cache).with_suffix(".json").write_text(json.dumps(best, ensure_ascii=False), encoding="utf-8")
    return best, trials
