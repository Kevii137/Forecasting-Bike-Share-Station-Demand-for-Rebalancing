"""LightGBM settings and fitting, shared by tuning, training and retraining.

Kept free of Ray so the retraining service can fit models without it.
"""
import lightgbm as lgb

from bikeshare.modeling.dataset import FEATURE_COLUMNS


def lightgbm_params(config: dict, threads: int, seed: int) -> dict:
    """Fixed settings plus the tuned ones. deterministic makes reruns repeatable."""
    return {
        "objective": "poisson", "metric": "poisson", "bagging_freq": 1,
        "num_threads": threads, "seed": seed, "deterministic": True,
        "force_row_wise": True, "verbosity": -1, **config,
    }


def fit(config: dict, train: tuple, validation: tuple, threads: int, seed: int,
        max_rounds: int, early_stopping_rounds: int) -> lgb.Booster:
    train_set = lgb.Dataset(train[0], train[1], feature_name=FEATURE_COLUMNS)
    valid_set = lgb.Dataset(validation[0], validation[1], reference=train_set)
    return lgb.train(
        lightgbm_params(config, threads, seed), train_set, num_boost_round=max_rounds,
        valid_sets=[valid_set], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
