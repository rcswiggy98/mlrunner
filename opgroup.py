import dataclasses
from dataclasses import dataclass
import functools
import threading
from typing import ItemsView
import pandas as pd
import numpy as np
import uuid
import optuna
import enum
import itertools
import functools

class CVType(enum.Enum):
    NESTED = 0
    CV_TRAIN = 1
    NO_CV = 2

def unique_id():
    return uuid.uuid4().hex

def _maybe_dataclass(cls, mapping, default=False):
    try:
        return cls(**mapping)
    except Exception:
        if not(default is False):
            return default
        return mapping

def _add_col(colname:str, value=None):
    def inner(df):
        if callable(value):
            df[colname] = value()
        else:
            df[colname] = value
        return df
    return inner

# three cases
# nested-cv (most complicated) 
# n outer folds, k inner folds each (each of these)
# non-nested CV 
# n outer folds serving
# no-CV

# for the training phase, you need to finish all trials
# or if you find one that encounters a stopping condition
# training phase 
# - reporting synchronization
# - trial.report -> block until a threading.Event() is released
# - tell synchronization (final value)
# - hyperparameter caching
# n optuna studies need to be initialized

# validation phase (can possibly skip?)
# choose the best parameters out of all the trials completed
# and compute the metric. n outer folds, no inners
# call these valsets. For nested/CV_VALID, it's the outer folds. 
# Each valset should link with a synchronization device.

# opset (probably). What you use to do the hyperparameter selection
# nested CV, this is just all the inner training folds associated with an outer fold 
# normal CV. It depends, either the folds will be used as the opset, or the
# implied holdout will be used. Can be configured. If the former, then
# we will need to synchronize. If the latter, the only CV step is the
# final validation phase.

# is there a better way to sychronize? Use dask-events, name the opsets.
# What is the primary key here?

# ASSUMPTION: there is only one trainable function that the user needs
# to implement. trial.report will be disabled in the validation phase
# trial.tell will just write the final value to some store to possibly
# reduce in the nested case or write directly as the estimated generalization 
# score in the client-facing result.estimated_objective_value

# NEED TO CODE fastai plugin for this.
# TODO: The user might want a different final metric to be computed then the one
# that is being used to optimized. I.e. the validation phase might need to be
# configured. Just provide a flag trial.should_validate. 
# non-nested CV
# no-CV (most preferred for trainables that will take a long time to execute)

# HACKY: OR implement __getitem__ for this, so you can wrap it in scikit learn?
@dataclass
class CVConfig:
    num_folds: int = None
    reduction: 'str | Callable' = 'mean'
    optimize_on_holdout: bool = False # has no effect when passed as inner_cv_config

# interaction with dask: provide an existing scheduler address or just
class OptimizationGroup(object):
    def __init__(self, trainable, study_wrapper, hparams=None, cv_config=None, 
                 inner_cv_config=None, _cache=None): 
        self.trainable = trainable
        # extract info from this
        self.n_trials = study_wrapper.n_trials
        self.study = study_wrapper.create_study()
        self.hparam_dict = hparams
        self.cv_config = _maybe_dataclass(CVConfig, cv_config)
        self.inner_cv_config = _maybe_dataclass(CVConfig, inner_cv_config)
        self._cache = _cache

        # WHAT IS THE INDEX for the job table?
        # trial_id, is_validating, fold, inner_fold, lock_idx
        # could just make two job tables.
        # in the `run` method of the executor, the first
        # futures are the training jobs, we wait for
        # all of those to execute. Then the validation jobs.
        # for nested CV though, we don't need to wait for all of them.
        self._init_job_table()

        # synchronization and communication primitives
        self._init_job_comms() 

        self._init_opset_table()
        self._init_opset_comms()

        # do we need a report thread, when we can build-in all of the
        # synchronization in dask? We just need to construct the dependencies
        # in dask. Probably a lot of wrapper code, and we would need to be
        # writing to some shared result store anyway because of the IPC we need to do.
        # stick to actor model for now.
        self._init_handler_thread() # listens to dask workers

    def _init_job_table(self):
        trials = list(range(self.n_trials))

        # first make the training jobs
        folds = self.folds(is_validating=False) 
        inner_folds = self.inner_folds(is_validating=False) 
        job_specs = itertools.product(trials, folds, inner_folds)
        train_job_table = pd.DataFrame.from_records(
            dict(trial=tid, is_validating=False, fold=i_fold, inner_fold=i_inner_fold)
            for tid, i_fold, i_inner_fold in job_specs
        )
        train_job_table['valset'] = None
        train_job_table = train_job_table.groupby(['trial', 'fold']).apply(
            _add_col('opset', value=unique_id)
        )
        
        folds = self.folds(is_validating=True) 
        inner_folds = self.inner_folds(is_validating=True) 
        val_job_specs = itertools.product(trials, folds, inner_folds)
        val_job_table = pd.DataFrame.from_records(
            dict(trial=tid, is_validating=True, fold=i_fold, inner_fold=i_inner_fold)
            for tid, i_fold, i_inner_fold in val_job_specs
        )
        jobs = pd.concat((train_job_table, val_job_table), axis=0)

        # do a groupby here to get the opset

        # to get the valsets, again a groupby

        # opsets should have an associated valset.


    def _init_job_comms(self):
        # mainly to setup kill events.
        pass

    def _init_tinfo_table(self):
        pass

    def _init_handler_thread(self):
        pass

    def folds(self, is_validating):
        if is_validating:
            if self.cv_type is CVType.NO_CV:
                return [None]
            else:
                return list(range(self.cv_config.num_folds)) 

        # reverse for training
        if self.cv_type is not CVType.NO_CV:
            return list(range(self.cv_config.num_folds)) 
        else:
            return [None]
        
    def inner_folds(self, is_validating):
        if is_validating and self.cv_type is CVType.NESTED:
            return list(range(self.inner_cv_config.num_folds))
        return [None]

    @property
    def cv_type(self):
        if self.inner_cv_config.num_folds > 1 and self.cv_config.num_folds > 1:
            return CVType.NESTED
        if self.cv_config.num_folds > 1:
            if self.cv_config.optimize_on_holdout:
                return CVType.CV_VALID
            return CVType.CV_TRAIN
        return CVType.NO_CV

    def restore(self):
        pass

    def __reduce__(self) -> 'str | Tuple[Any, ...]':
        return super().__reduce__()
