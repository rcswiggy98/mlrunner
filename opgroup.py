import dataclasses
from dataclasses import dataclass
import functools
import threading
from typing import ItemsView, NamedTuple, Type, Union
import pandas as pd
import numpy as np
import uuid
import optuna
import enum
import itertools
import functools

import dask.distributed as dist

NULL = object()

class CVType(enum.Enum):
    # assumes you only have one dataset. Optimize hyperparameters using cross validation
    # on inner training sets independently, and then combine in an outer cross validation
    # to estimate generalization error.
    NESTED = 0 
    # assumes you have two datasets. optimize hyperparameters using 
    # cross validation on the training set, then estimate generalization error on test set.
    CV_TRAIN = 1 
    # assumes you have three datasets. train on the training set
    # optimize hparams wrt to valset, and estimate generalization error on test set.
    NO_CV = 2

class JobStatus(enum.Enum):
    QUEUED = 0
    RUNNING = 1
    NEEDS_REPORT = 2
    COMPLETED = 3
    ERROR = 4
    NO_TELL_TRY_REPORT = 5

def unique_id() -> str:
    return uuid.uuid4().hex

def _add_col(colname:str, value=None):
    def inner(df):
        if callable(value):
            df[colname] = value(df)
        else:
            df[colname] = value
        return df
    return inner

# def _add_row(df: pd.DataFrame, row: Union[dict, pd.Series]):
#     if isinstance(row, dict):
#         assert all(k in df.columns for k in row.keys()) and len(row) == len(df.columns)
#         new_row = pd.Series([row[c] for c in df.columns], index=df.columns.tolist())
#     else:
#         new_row = row
#     new_df = pd.concat((df, new_row), axis=0, ignore_index=False)
#     return new_df

def _maybe_dataclass(dcls, mapping, default=False):
    try:
        return dcls(**mapping)
    except Exception:
        if not(default is False):
            return default
        return mapping

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

# interaction with dask: provide an existing scheduler address or jus
# report perf of parameters in descending or ascending order.
class OptimizationGroup(object):
    def __init__(self, trainable, study_wrapper, hparams=None, cv_config=None, 
                 inner_cv_config=None, _cache=None): 
        self._id = unique_id()
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
        # batching also applies here, we can make the trials independent as well.
        # i.e. trials 0, 1, 2 are completely independent of each other.
        opset_key = ['fold', 'trial'] if self.cv_type is CVType.NESTED else 'is_validating'
        train_job_table = train_job_table.groupby(opset_key).apply(
            _add_col('opset', value=lambda df: df['fold'].values[0])
        )

        flatten_col = ('inner_fold', 'fold', NULL)[self.cv_type.value]
        folds = self.folds(is_validating=True) 
        inner_folds = self.inner_folds(is_validating=True) 
        # add one val job spec to each opset.
        job_table = train_job_table.groupby(opset_key).apply(
            lambda df: df.append(
                {
                    **{k:(v if k != flatten_col else None) for k,v in df.iloc[0].to_dict().items()}, 
                    **{'is_validating': True}
                }
            )
        )
        self.job_table = job_table

        # EXAMPLES 
        # NESTED
        # t, f, i, val   opset
        # 0, 0, 0, False  A
        # 0, 0, 1, False  A
        # 0, 0, 2, False  A
        # 0, 1, 0, False  B
        # 0, 1, 1, False  B
        # 0, 1, 2, False  B
        # 1, 0, 0, False  AA
        # 1, 0, 1, False  AA
        # 1, 0, 2, False  AA
        # 1, 1, 0, False  BB
        # 1, 1, 1, False  BB
        # 1, 1, 2, False  BB

        # 0, 0, None, True A
        # 0, 1, None, True B
        # 1, 0, None, True AA
        # 1, 1, None, True BB

        # final stage, need to reduce on all val, gby trial.
        
        # Non-NESTED (holdout is used to optimize hyperparameters)
        # t, f, i,    val   opset
        # 0, 0, None, False  A
        # 0, 1, None, False  A
        # 1, 0, None, False  A
        # 1, 1, None, False  A
        # 0, None, None, True, A
        # 1, None, None, True, A

        # No-CV
        # t, f, i, val, opset
        # 0, None, None, False, A
        # 0, None, None, True, A
        # 1, None, None, False, A
        # 1, None, None, True, A

    def _kill_event_name(self, idx):
        return f"{self._id}_kill_job_{idx}"

    def _init_job_comms(self):
        # may need to setup a local client.
        # may just make it local to the worker (threading.Event)
        self.job_kill_events = {
            idx: dist.Event(name=self._kill_event_name(idx), 
                            client=None)
            for idx in self.job_table.index
        }
        
        # one per opgroup and trial? or just opgroup?
        # for nested CV, the studies are independent...
        self.report_wakeup_events = {
            opgroup: threading.Event()
            for opgroup in self.job_table['opgroup'].unique()
        }

        # one per trial
        self.tell_sync_events = {
            trial_id: threading.Event()
            for trial_id in self.job_table['trial_id'].unique()
        }

        # mutable state that must be synchronized if the report thread and
        # workers will be updating
        self._report_values = {
            idx: NULL
            for idx in self.job_table.index
        }
        self.report_lock = threading.Lock()

        self._job_statuses = {
            idx: JobStatus.QUEUED
            for idx in self.job_table.index
        }
        self.status_lock = threading.Lock()

    def _init_opset_table(self):
        # trial_id opset study_id
        pass

    def _init_handler_thread(self):
        # iterate over every opset, 
        # - report to the studies for each opgroup after synchronizing results, 
        #   block the workers until so, and release them when ready
        # - check the health of the jobs, cancel the opset entirely if there's an
        #   unhandled exception in one of the folds
        # - parse the final return value, or final set of values thru trial.tell
        pass

    def folds(self, is_validating: bool):
        if is_validating:
            if self.cv_type is CVType.NO_CV:
                return [None]
            return list(range(self.cv_config.num_folds)) 

        # reverse for training
        if self.cv_type is not CVType.NO_CV:
            return list(range(self.cv_config.num_folds)) 
        return [None]
        
    def inner_folds(self, is_validating):
        if not is_validating and self.cv_type is CVType.NESTED:
            return list(range(self.inner_cv_config.num_folds))
        return [None]

    @property
    def cv_type(self):
        if self.inner_cv_config.num_folds > 1 and self.cv_config.num_folds > 1:
            return CVType.NESTED
        if self.cv_config.num_folds > 1:
            return CVType.CV_TRAIN
        return CVType.NO_CV

    def restore(self):
        pass

    def __reduce__(self) -> 'str | Tuple[Any, ...]':
        return super().__reduce__()
