import multiprocessing as mp
import optuna
from mlrunner.session import Cache

def _study_manager_thread():
    pass

class Trial(object):
    """
    Wrapper around an optuna trial containing additional trial-local data, and
    a pointer to the session's global Cache, which lives on a dask Actor.
    Overloads __getattr__ to perform 
    - sampling of hyperparameters using Optuna's suggest optimization algorithms.
    - fixed local parameter access.
    - remote parameter access for Cached values (large datasets that are expensive to load/preprocess)

    Also facilitates reporting directly to optuna.

    This is passed along to the user trainable. 
    """
    def __init__(self, trial: optuna.trial.Trial, hypers: dict, params: dict, cache: Cache) -> None:
        self._trial = trial
        self._hypers = hypers
        self._params = params
        self._cache = cache