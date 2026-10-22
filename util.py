from dataclasses import dataclass
import numpy as np

INPUT_KEYS = ('recent_data', 'day_data', 'week_data', 'month_data')


def _integer(value, name, minimum=0):
    if not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


@dataclass(frozen=True)
class WindowSpec:
    alpha: int = 6
    beta: int = 2
    gamma: int = 2
    xi: float = 1.0
    horizon: int = 12
    points_per_day: int = 288
    month_days: int = 30
    month_reference: str = 'fixed_days'

    def __post_init__(self):
        for name in ('alpha', 'beta', 'gamma'):
            _integer(getattr(self, name), name)
        for name in ('horizon', 'points_per_day', 'month_days'):
            _integer(getattr(self, name), name, 1)
        _integer(2 * self.xi, '2*xi')
        if self.month_reference not in ('fixed_days', 'calendar_month'):
            raise ValueError('Unknown monthly reference convention.')
        if self.beta > self.points_per_day or self.gamma > 7*self.points_per_day:
            raise ValueError('A historical input extends beyond the current observation.')
        minimum_lag = (self.month_days if self.month_reference == 'fixed_days' else 28)*self.points_per_day
        if 2*self.xi > minimum_lag:
            raise ValueError('Monthly half-width may extend beyond the current observation.')

    @property
    def window_lengths(self):
        return dict(recent_data=self.alpha+1, day_data=2*self.beta+1,
                    week_data=2*self.gamma+1, month_data=int(4*self.xi+1))

    @property
    def offsets(self):
        if self.month_reference != 'fixed_days':
            raise ValueError('Calendar months do not have a constant offset; use input_indices().')
        return {
            'recent_data': np.arange(-self.alpha, 1, dtype=np.int64),
            'day_data': np.arange(-self.beta, self.beta+1, dtype=np.int64)-self.points_per_day,
            'week_data': np.arange(-self.gamma, self.gamma+1, dtype=np.int64)-7*self.points_per_day,
            'month_data': np.arange(-int(2*self.xi), int(2*self.xi)+1, dtype=np.int64)
                          - self.month_days*self.points_per_day,
        }

    @property
    def warmup(self):
        return -min(int(x.min()) for x in self.offsets.values())

    def input_indices(self, origin, month_centers=None):
        t = np.asarray(origin, dtype=np.int64)
        if self.month_reference == 'fixed_days':
            return {key: t[..., None]+off for key, off in self.offsets.items()}
        if month_centers is None:
            raise ValueError('Calendar-month windows require timestamp-based month centers.')
        center = np.asarray(month_centers)[t]
        if (center < 0).any():
            raise ValueError('No previous-calendar-month observation is available.')
        return {
            'recent_data': t[..., None]+np.arange(-self.alpha, 1),
            'day_data': t[..., None]-self.points_per_day+np.arange(-self.beta, self.beta+1),
            'week_data': t[..., None]-7*self.points_per_day+np.arange(-self.gamma, self.gamma+1),
            'month_data': center[..., None]+np.arange(-int(2*self.xi), int(2*self.xi)+1),
        }

    def origins(self, length, label_start=0, label_end=None, month_centers=None):
        end = length if label_end is None else label_end
        if not 0 <= label_start <= end <= length:
            raise ValueError('Invalid label interval.')
        if self.month_reference == 'fixed_days':
            start = self.warmup
        else:
            if month_centers is None or len(month_centers) != length:
                raise ValueError('Calendar-month centers must match the traffic record count.')
            start = max(self.alpha, self.points_per_day+self.beta, 7*self.points_per_day+self.gamma)
        result = np.arange(max(start, label_start-1), end-self.horizon, dtype=np.int64)
        if self.month_reference == 'calendar_month':
            center = np.asarray(month_centers)[result]
            half = int(2*self.xi)
            result = result[(center >= half) & (center+half <= result)]
        return result


def validate_timestamps(timestamps, length):
    timestamps = np.asarray(timestamps).astype('datetime64[ns]')
    if timestamps.shape != (length,) or np.isnat(timestamps).any():
        raise ValueError('Timestamps must be a nonmissing vector matching the record count.')
    if length > 1 and not (np.diff(timestamps) == np.timedelta64(5, 'm')).all():
        raise ValueError('Timestamps must be continuous five-minute observations; do not remove time rows.')
    return timestamps


def calendar_month_centers(timestamps):
    timestamps = validate_timestamps(timestamps, len(timestamps))
    day = timestamps.astype('datetime64[D]')
    month = timestamps.astype('datetime64[M]')
    month_start = month.astype('datetime64[D]')
    previous_start = (month-np.timedelta64(1, 'M')).astype('datetime64[D]')
    previous_last = month_start-np.timedelta64(1, 'D')
    desired_day = previous_start + np.minimum(day-month_start, previous_last-previous_start)
    desired = desired_day.astype('datetime64[ns]')+(timestamps-day.astype('datetime64[ns]'))
    found = np.searchsorted(timestamps, desired)
    valid = found < len(timestamps)
    valid[valid] &= timestamps[found[valid]] == desired[valid]
    return np.where(valid, found, -1).astype(np.int64)


def validate_data(data, flow_index=0, allow_missing=False, expected_features=None):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or min(data.shape) < 1:
        raise ValueError('Traffic data must have shape [time, nodes, features].')
    if not 0 <= flow_index < data.shape[-1]:
        raise ValueError('flow_index is outside the feature dimension.')
    if expected_features is not None and data.shape[-1] != expected_features:
        raise ValueError('The manuscript requires three input channels: flow, speed and occupancy. '
                         'Missing physical variables cannot be fabricated.')
    if np.isinf(data).any() or (not allow_missing and np.isnan(data).any()):
        raise ValueError('Traffic data contain unsupported NaN/Inf values.')
    return data


def extract_sample(data, origin, spec, flow_index=0, month_centers=None):
    if origin < 0 or origin+spec.horizon >= len(data):
        raise ValueError('Insufficient observations or future labels at this forecast origin.')
    indices = spec.input_indices(origin, month_centers)
    if any((idx < 0).any() or (idx > origin).any() for idx in indices.values()):
        raise ValueError('Input windows must lie in the observed past, including the current sample.')
    sample = {key: data[idx] for key, idx in indices.items()}
    sample['output_data'] = data[origin+1:origin+spec.horizon+1, :, flow_index:flow_index+1]
    return sample


def prepare_data_new(data, spec=None, flow_index=0, month_centers=None):
    spec = spec or WindowSpec()
    data = validate_data(data, flow_index)
    origins = spec.origins(len(data), month_centers=month_centers)
    if not len(origins):
        raise ValueError('Not enough records for monthly history and forecast horizon.')
    samples = [extract_sample(data, t, spec, flow_index, month_centers) for t in origins]
    return {key: np.stack([sample[key] for sample in samples]) for key in (*INPUT_KEYS, 'output_data')}


def split_origins(length, spec, train_ratio=0.7, val_ratio=0.1, month_centers=None):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError('Training, validation and test ratios must all be positive.')
    train_end = int(length*train_ratio)
    val_end = train_end + int(length*val_ratio)
    bounds = ((0, train_end), (train_end, val_end), (val_end, length))
    splits = tuple(spec.origins(length, a, b, month_centers) for a, b in bounds)
    if any(not len(s) for s in splits):
        raise ValueError('A split is empty after allowing for monthly history and forecast horizon.')
    return splits, (train_end, val_end)


def interpolate_short_gaps(data, max_gap=1, flow_index=0):
    _integer(max_gap, 'max_gap')
    raw = validate_data(data, flow_index, allow_missing=True)
    filled = raw.copy()
    available_after = np.arange(len(raw), dtype=np.int64)
    observed_targets = np.isfinite(raw[:, :, flow_index]).all(axis=1)
    count = 0
    for node in range(raw.shape[1]):
        for feature in range(raw.shape[2]):
            values = filled[:, node, feature]
            missing = np.isnan(values)
            changes = np.diff(np.r_[False, missing, False].astype(np.int8))
            for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                if end-start <= max_gap and start > 0 and end < len(raw):
                    values[start:end] = np.linspace(values[start-1], values[end], end-start+2)[1:-1]
                    available_after[start:end] = np.maximum(available_after[start:end], end)
                    count += end-start
    return filled, available_after, observed_targets, {
        'max_interpolation_gap': int(max_gap), 'filled_values': int(count),
        'unfilled_values': int(np.isnan(filled).sum()),
        'target_policy': 'exclude windows with any originally missing flow labels',
        'input_policy': 'filled values available only after right interpolation endpoint is observed',
    }


def filter_valid_origins(origins, data, spec, available_after, observed_targets, month_centers=None):
    origins = np.asarray(origins, dtype=np.int64)
    good_frame = np.isfinite(data).all(axis=(1, 2))
    valid = np.ones(len(origins), dtype=bool)
    for indices in spec.input_indices(origins, month_centers).values():
        valid &= good_frame[indices].all(axis=1)
        valid &= (available_after[indices] <= origins[:, None]).all(axis=1)
    labels = origins[:, None]+np.arange(1, spec.horizon+1)
    valid &= np.asarray(observed_targets)[labels].all(axis=1)
    return origins[valid]


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, allow_missing=False):
        values = np.asarray(values, dtype=np.float64)
        if not values.size or np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
            raise ValueError('Scaler requires usable training observations.')
        flat = values.reshape(-1, values.shape[-1])
        if np.isnan(flat).all(axis=0).any():
            raise ValueError('At least one observed training value is required for every channel.')
        mean = np.nanmean(flat, axis=0) if allow_missing else flat.mean(axis=0)
        scale = np.nanstd(flat, axis=0) if allow_missing else flat.std(axis=0)
        return cls(mean, np.where(scale > 1e-12, scale, 1.0))

    def transform(self, values):
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale + self.mean

    def state_dict(self):
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(np.asarray(state['mean']), np.asarray(state['scale']))


def normalize_dataset_new(train, val, test, return_input_scaler=False):
    input_scaler = FeatureScaler.fit(np.concatenate([
        train[k].reshape(-1, train[k].shape[-1]) for k in INPUT_KEYS
    ]))
    output_scaler = FeatureScaler.fit(train['output_data'])
    normalized = tuple({k: (output_scaler if k == 'output_data' else input_scaler).transform(v)
                        for k, v in part.items()} for part in (train, val, test))
    result = (*normalized, output_scaler)
    return (*result, input_scaler) if return_input_scaler else result


def build_ncsm(adjacency, distances, chi=0.1, degree_mode='neighbors'):
    a, d = np.asarray(adjacency), np.asarray(distances, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or d.shape != a.shape:
        raise ValueError('Adjacency and distance must be square matrices of the same shape.')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Adjacency must contain finite nonnegative values.')
    if not np.isfinite(chi) or not 0 < chi < 1:
        raise ValueError('chi must be strictly between 0 and 1.')
    connected = a > 0
    np.fill_diagonal(connected, False)
    selected = d[connected]
    if not len(selected) or not np.isfinite(selected).all() or (selected <= 0).any():
        raise ValueError('Every connected nonself edge needs a positive finite physical distance.')
    if degree_mode == 'neighbors':
        degree = (connected | connected.T).sum(axis=0, dtype=np.float32)[:, None]
    elif degree_mode in ('in', 'out'):
        degree = connected.sum(axis=0 if degree_mode == 'in' else 1, dtype=np.float32)[:, None]
    else:
        raise ValueError('degree_mode must be neighbors, in or out.')
    weights = np.zeros_like(d)
    weights[connected] = 1.0 / selected
    np.fill_diagonal(weights, 1.0 / (chi * selected.min()))
    weights /= weights.max()
    source, target = np.nonzero(weights)
    edge_index = np.stack((source, target)).astype(np.int64)
    return edge_index, weights[source, target, None].astype(np.float32), degree, weights.astype(np.float32)


def node_distance_to_edge_attr_(node_distance, adjacency, chi=0.1):
    return build_ncsm(adjacency, node_distance, chi)[1]


def calculate_metrics(actuals, predictions, mape_threshold=0.0):
    y, p = np.asarray(actuals, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    if y.shape != p.shape or not y.size:
        raise ValueError('Actuals and predictions must have the same nonempty shape.')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Evaluation contains NaN/Inf.')
    if not np.isfinite(mape_threshold) or mape_threshold < 0:
        raise ValueError('mape_threshold must be finite and nonnegative.')
    error = np.abs(y-p)
    valid = np.abs(y) > mape_threshold
    mape = float(100*np.mean(error[valid]/np.abs(y[valid]))) if valid.any() else float('nan')
    denominator = np.abs(y).sum()
    wape = float(100*error.sum()/denominator) if denominator else float('nan')
    return float(error.mean()), float(np.sqrt(np.mean(error**2))), mape, wape


def metric(pred, label):
    return calculate_metrics(label, pred)[:3]


def forecast_metrics(actuals, predictions, mape_threshold=0.0):
    if np.asarray(actuals).ndim != 4:
        raise ValueError('Expected [samples, horizon, nodes, 1].')
    names = ('MAE', 'RMSE', 'MAPE_percent', 'WAPE_percent')
    rows = [dict(zip(names, calculate_metrics(actuals[:, h], predictions[:, h], mape_threshold)))
            for h in range(actuals.shape[1])]
    return {
        'per_horizon': [dict(step=i+1, **row) for i, row in enumerate(rows)],
        'mean_over_horizons': {k: float(np.mean([r[k] for r in rows])) for k in names},
        'pooled': dict(zip(names, calculate_metrics(actuals, predictions, mape_threshold))),
    }
