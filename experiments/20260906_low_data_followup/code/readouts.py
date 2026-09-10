"""Six preregistered training-only stable readouts; no upstream model access.

The callable contract matches low_data_models.fit_predict. Model states contain
plain dictionaries and original canonical torch/sklearn classes, so arbitrary
importlib loading of this file does not break trusted-local pickle replay.
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from torch import nn

from nesso_pxr import low_data_models as original

METHODS = (
    "repr_ridge_unscaled", "repr_ridge_stable", "repr_mlp_unscaled",
    "repr_mlp_stable", "repr_mlp_stable_centered",
    "repr_mlp_stable_centered_regularized",
)
RIDGE_METHODS = METHODS[:2]
SCALING_C = 0.1
SCALING_EPSILON = 1e-8
REGULARIZED_WEIGHT_DECAY = 0.01


def quantiles(x):
    a = np.asarray(x, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    return dict(count=int(a.size), minimum=float(a.min()), maximum=float(a.max()),
                quantiles=np.quantile(a, [0, .01, .25, .5, .75, .99, 1]).tolist())


def fit_preprocessing(x, w, scaled):
    """Pure training fit. Global RMS is sqrt(mean(column variances))."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    if x.ndim != 2 or not len(x) or w.shape != (len(x),):
        raise ValueError("nonempty aligned preprocessing arrays required")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("finite features and positive weights required")
    if not scaled:
        return {"scaled": False, "mean": np.zeros(x.shape[1]),
                "scale": np.ones(x.shape[1]), "global_spread": None}
    mean = np.average(x, axis=0, weights=w)
    var = np.average(np.square(x - mean), axis=0, weights=w)
    global_spread = float(np.sqrt(var.mean()))
    scale = np.maximum(np.sqrt(var), max(SCALING_C * global_spread, SCALING_EPSILON))
    return dict(scaled=True, mean=mean, scale=scale, global_spread=global_spread)


def transform(x, preprocessing):
    return (np.asarray(x, dtype=float) - preprocessing['mean']) / preprocessing['scale']


def _predict_one(state, x):
    z = transform(x, state['preprocessing'])
    if state['kind'] == 'ridge':
        return state['model'].predict(z).astype(float)
    state['model'].eval()
    z = torch.as_tensor(z, dtype=torch.float32)
    with torch.inference_mode():
        pred = state['model'](z[:, :384], z[:, 384:])[2].numpy().astype(float)
    return pred + state['target_offset']


def predict_model(model_state, features, *, batch_size=1024):
    """Label/checkpoint-free per-seed replay, bounded CPU and batch intermediates."""
    if model_state.get('schema_version') != 'stable_readouts_v1' or model_state.get('method') not in METHODS:
        raise ValueError('unsupported stable readout model state')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    f, n = original._validate_features(features)
    x = original._matrix(f, 'repr_mlp')
    states = model_state['models']
    if not states or len(states) != len(model_state['seeds']):
        raise ValueError('one saved model per seed required')
    p = np.empty((len(states), n), dtype=float)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):
            for start in range(0, n, batch_size):
                for i, state in enumerate(states):
                    p[i, start:start+batch_size] = _predict_one(state, x[start:start+batch_size])
    finally:
        torch.set_num_threads(previous)
    if not np.isfinite(p).all():
        raise FloatingPointError('nonfinite unclipped replay predictions')
    return dict(pred=p.mean(axis=0), ensemble_std=p.std(axis=0), seed_predictions=p)


def _fit(method, x, y, w, candidate, epochs, seed, config, validation=None):
    preprocessing = fit_preprocessing(x, w, 'unscaled' not in method)
    offset = float(np.average(y, weights=w)) if 'centered' in method else 0.0
    z = transform(x, preprocessing)
    diagnostics = dict(global_spread=preprocessing['global_spread'],
                       scales=quantiles(preprocessing['scale']),
                       transformed_training=quantiles(z), target_offset=offset)
    if method in RIDGE_METHODS:
        model = Ridge(alpha=float(candidate), fit_intercept=True, solver='svd')
        model.fit(z, y, sample_weight=w/w.mean())
        state = dict(kind='ridge', model=model, preprocessing=preprocessing, target_offset=0.0)
        predictions = [] if validation is None else [_predict_one(state, validation)]
        return state, predictions, [], diagnostics
    model = original.make_neural_model('repr_mlp', config, None, seed=seed)
    z = torch.as_tensor(z, dtype=torch.float32)
    target = torch.as_tensor(y-offset, dtype=torch.float32)
    weight = torch.as_tensor(w/w.mean(), dtype=torch.float32)
    decay = REGULARIZED_WEIGHT_DECAY if method.endswith('_regularized') else 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(candidate), weight_decay=decay)
    generator = torch.Generator().manual_seed(int(seed))
    state = dict(kind='neural', model=model, preprocessing=preprocessing, target_offset=offset)
    predictions, losses = [], []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for ix in torch.randperm(len(y), generator=generator).split(int(config.get('batch_size', 64))):
            optimizer.zero_grad(set_to_none=True)
            loss = original.weighted_smooth_l1(model(z[ix, :384], z[ix, 384:]), target[ix], weight[ix], beta=.5)
            loss = loss * weight[ix].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite neural training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
        losses.append(total_loss/len(y))
        if validation is not None:
            predictions.append(_predict_one(state, validation))
    model.eval()
    return state, predictions, losses, diagnostics


def fit_predict(method, train_features, y_train, assay_se_train, test_features, *,
                config, checkpoint_path, seeds=(42, 43, 44)):
    """Select LR/epoch or alpha within fit labels; checkpoint_path is unused.

    Uses config se_floor/inner_folds/grids/epochs/batch_size as original API.
    Stable constants and weight decay are fixed module-level recipe choices.
    Unknown feature keys (including labels/descriptors) fail closed.
    """
    start = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f'unknown stable method: {method}')
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and unique')
    y, se = np.asarray(y_train, dtype=float), np.asarray(assay_se_train, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or se.shape != y.shape:
        raise ValueError('finite aligned training labels/errors required')
    w = original._weights(se, float(config['se_floor']))
    train, _ = original._validate_features(train_features, expected=len(y))
    test, _ = original._validate_features(test_features)
    x = original._matrix(train, 'repr_mlp')
    xt = original._matrix(test, 'repr_mlp')
    neural = method not in RIDGE_METHODS
    epochs = int(config.get('epochs', 40)) if neural else 1
    if epochs < 1 or int(config.get('batch_size', 64)) < 1:
        raise ValueError('positive epochs/batch size required')
    if float(config.get('neural_huber_beta', .5)) != .5:
        raise ValueError('fixed smooth-L1 beta=.5')
    key = 'neural_learning_rate' if neural else 'ridge_alpha'
    grid = config['grids'][key]
    if not grid or any(not np.isfinite(v) or v <= 0 for v in grid):
        raise ValueError('positive finite candidate grid required')
    folds, fold_audit = original._folds(train, len(y), int(config['inner_folds']), seeds[0])
    selection = dict(**fold_audit, metric='pooled_inner_oof_assay_weighted_mae', selection_seed=seeds[0], candidates=[])
    audits, history, states = [], [], []
    best = np.inf
    chosen, chosen_epochs = None, None
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):
            for value in grid:
                oof = np.full((epochs, len(y)), np.nan)
                for fold, (a, b) in enumerate(folds):
                    _, prediction, _, diagnostics = _fit(method, x[a], y[a], w[a], value, epochs, seeds[0], config, x[b])
                    oof[:, b] = prediction
                    audits.append(dict(scope='inner_train', fold=fold, candidate=float(value),
                                       train_indices=a.tolist(), validation_indices=b.tolist(), **diagnostics))
                scores = [original._weighted_mae(y, p, w) for p in oof]
                epoch_index = int(np.argmin(scores))
                selection['candidates'].append(dict(value=float(value), epoch_weighted_mae=scores, best_epoch=epoch_index+1))
                if scores[epoch_index] < best:
                    best, chosen, chosen_epochs = scores[epoch_index], value, epoch_index+1
            selection['selected'] = {key: float(chosen)}
            if neural:
                selection['selected']['epochs'] = chosen_epochs
            for seed in (seeds if neural else seeds[:1]):
                state, _, losses, diagnostics = _fit(method, x, y, w, chosen, chosen_epochs, seed, config)
                audits.append(dict(scope='final_fit', seed=seed, train_indices=list(range(len(y))), **diagnostics))
                history.append(dict(seed=seed, epoch_training_loss=losses))
                states.append(state)
            if not neural:
                states = states * len(seeds)
            artifact = dict(schema_version='stable_readouts_v1', method=method, seeds=seeds,
                            models=states, threads=2, selection=selection)
            prediction = predict_model(artifact, test)
            reloaded = predict_model(pickle.loads(pickle.dumps(artifact)), test)
            if not np.array_equal(prediction['seed_predictions'], reloaded['seed_predictions']):
                raise AssertionError('same-batch serialization replay failed')
            transformed_test = quantiles(transform(xt, states[0]['preprocessing']))
    finally:
        torch.set_num_threads(previous)
    count, _ = original._parameter_count(states[0])
    return dict(**prediction, model_state=artifact, selection=selection, n_parameters=count,
                fit_seconds=time.perf_counter()-start,
                fit_audit=dict(n_fit=len(y), seed_ids=list(seeds), threads=2,
                    preprocessing_fits=audits, final_fit_history=history,
                    transformed_evaluation=transformed_test, prediction_distribution=quantiles(prediction['pred']),
                    serialization_verified=True, serialization_format='trusted-local pickle',
                    identical_non_neural_seed_replicas=not neural,
                    upstream_representation_frozen=True, calibration_labels_accessed=False,
                    test_labels_accessed=False, output_clipping=False,
                    weight_decay=REGULARIZED_WEIGHT_DECAY if method.endswith('_regularized') else 0.0))
