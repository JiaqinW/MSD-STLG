import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MSDSTLG
from util import (INPUT_KEYS, FeatureScaler, WindowSpec, build_ncsm, extract_sample,
                  forecast_metrics, split_origins, validate_data, validate_timestamps,
                  calendar_month_centers, interpolate_short_gaps, filter_valid_origins)


@dataclass
class Config:
    input_dim: int = 3
    hidden_dim: int = 16
    num_layers: int = 1
    degree_dim: int = 16
    out_channels: int = 56
    num_heads: int = 2
    decoderout: int = 20
    decoder_num_layers: int = 1
    out_length: int = 12


class TrafficDataset(Dataset):
    """Extract windows on demand, preserving temporal and node dimensions."""
    def __init__(self, data, origins, spec, input_scaler, output_scaler, flow_index=0,
                 month_centers=None):
        self.data, self.origins, self.spec = data, origins, spec
        self.input_scaler, self.output_scaler = input_scaler, output_scaler
        self.flow_index = flow_index
        self.month_centers = month_centers

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        sample = extract_sample(self.data, int(self.origins[index]), self.spec, self.flow_index,
                                self.month_centers)
        inputs = {k: torch.from_numpy(self.input_scaler.transform(sample[k])) for k in INPUT_KEYS}
        labels = torch.from_numpy(self.output_scaler.transform(sample['output_data']))
        return inputs, labels


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_matrix(path):
    path = Path(path)
    if path.suffix == '.npy':
        result = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(f'{path}: expected exactly one matrix in the archive.')
            result = archive[archive.files[0]]
    elif path.suffix in ('.pkl', '.pickle'):
        with path.open('rb') as stream:
            result = pickle.load(stream, encoding='latin1')
        if isinstance(result, (tuple, list)) and len(result) == 3 and isinstance(result[1], dict):
            result = result[2]
    else:
        raise ValueError('Graph files must be .npy, .npz or .pkl matrices.')
    if hasattr(result, 'toarray'):
        result = result.toarray()
    return np.asarray(result)


def synthetic_data(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(600, dtype=np.float32)[:, None]
    phase = np.arange(4, dtype=np.float32)[None]
    flow = 80 + 15*np.sin(t*2*np.pi/8+phase) + rng.normal(0, 1, (600, 4))
    data = np.stack((flow, 60-flow*0.1, flow*0.002), axis=-1).astype(np.float32)
    adjacency = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    distance = np.abs(np.arange(4)[:, None]-np.arange(4)[None, :]).astype(np.float32)
    return data, adjacency, distance


def move_inputs(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


def run_epoch(model, loader, graph, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, target in loader:
            inputs, target = move_inputs(inputs, device), target.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, *graph)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss encountered; training stopped.')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()*target.numel()
            count += target.numel()
    if not count:
        raise ValueError('Empty data loader.')
    return loss_sum/count


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def train_and_evaluate(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('epochs, batch-size and patience must be positive.')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning-rate must be positive.')
    if not np.isfinite(args.mape_threshold) or args.mape_threshold < 0:
        raise ValueError('mape-threshold must be finite and nonnegative.')
    if args.missing_value is not None and not np.isfinite(args.missing_value):
        raise ValueError('missing-value must be a finite sentinel; NaN is already treated as missing.')
    if any(getattr(args, k) < 1 for k in ('hidden_dim', 'degree_dim', 'out_channels', 'num_heads',
                                         'decoder_hidden', 'encoder_layers', 'decoder_layers')):
        raise ValueError('Model dimensions and layer counts must be positive.')
    seed_everything(args.seed)
    timestamps = None
    if args.synthetic_smoke:
        torch.set_num_threads(1)
        data, adjacency, distance = synthetic_data(args.seed)
        points_per_day = 8
    else:
        points_per_day = 288
        with np.load(args.data, allow_pickle=False) as archive:
            data = archive['data']
            if 'timestamps' in archive.files:
                timestamps = archive['timestamps']
        adjacency, distance = load_matrix(args.adjacency), load_matrix(args.distance)
        if args.timestamps:
            timestamps = np.load(args.timestamps, allow_pickle=False)
    if timestamps is not None:
        timestamps = validate_timestamps(timestamps, len(data))
    if args.synthetic_smoke and args.month_reference == 'calendar_month':
        raise ValueError('Compressed synthetic cycles cannot represent calendar months.')
    if args.month_reference == 'calendar_month' and timestamps is None:
        raise ValueError('Calendar-month extraction requires --timestamps or an NPZ timestamps field.')
    if args.missing_value is not None:
        data = np.asarray(data, dtype=np.float32).copy()
        data[data == args.missing_value] = np.nan
    raw_data = validate_data(data, args.flow_index, allow_missing=True, expected_features=3)
    if (raw_data < 0).any():
        raise ValueError('Raw flow, speed and occupancy must be nonnegative; declare a missing sentinel '
                         'or supply unstandardized physical observations.')
    data, available_after, observed_targets, preprocessing = interpolate_short_gaps(
        raw_data, args.max_interpolation_gap, args.flow_index)
    if adjacency.shape != (data.shape[1], data.shape[1]):
        raise ValueError('Graph dimensions do not match traffic data node count.')
    spec = WindowSpec(args.alpha, args.beta, args.gamma, args.xi,
                      args.horizon, points_per_day, args.month_days, args.month_reference)
    month_centers = calendar_month_centers(timestamps) if args.month_reference == 'calendar_month' else None
    origins, (train_end, val_end) = split_origins(len(data), spec, args.train_ratio,
                                               args.val_ratio, month_centers)
    candidate_sizes = [len(part) for part in origins]
    origins = tuple(filter_valid_origins(part, data, spec, available_after, observed_targets,
                                         month_centers) for part in origins)
    if any(not len(part) for part in origins):
        raise ValueError('A split has no valid windows after missing-data and causality checks.')
    input_scaler = FeatureScaler.fit(raw_data[:train_end], allow_missing=True)
    output_scaler = FeatureScaler.fit(raw_data[:train_end, :, args.flow_index:args.flow_index+1],
                                      allow_missing=True)
    datasets = [TrafficDataset(data, indices, spec, input_scaler, output_scaler, args.flow_index,
                               month_centers)
                for indices in origins]
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(part, batch_size=args.batch_size, shuffle=(i == 0),
                          drop_last=False, generator=generator if i == 0 else None)
               for i, part in enumerate(datasets)]
    edge_index, edge_weight, degrees, _ = build_ncsm(adjacency, distance, args.chi, args.degree_mode)
    if args.ncsm_prior == 'binary':
        edge_weight = np.ones_like(edge_weight)
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    graph = (torch.as_tensor(degrees, device=device),
             torch.as_tensor(edge_index, device=device),
             torch.as_tensor(edge_weight, device=device))
    config = Config(input_dim=data.shape[-1], hidden_dim=args.hidden_dim,
                    num_layers=args.encoder_layers, degree_dim=args.degree_dim,
                    out_channels=args.out_channels, num_heads=args.num_heads,
                    decoderout=args.decoder_hidden, decoder_num_layers=args.decoder_layers,
                    out_length=args.horizon)
    model = MSDSTLG(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output/'best_model.pt'
    if checkpoint_path.exists():
        raise FileExistsError(f'Choose a new --output-dir; checkpoint already exists: {checkpoint_path}')
    metadata = {
        'arguments': vars(args), 'model_config': asdict(config), 'window_spec': asdict(spec),
        'input_scaler': input_scaler.state_dict(), 'output_scaler': output_scaler.state_dict(),
        'split_boundaries': [train_end, val_end], 'split_sizes': [len(x) for x in origins],
        'parameters': sum(p.numel() for p in model.parameters()),
        'implementation_version': 3,
        'ncsm_fusion': 'alpha[j,i] = S[j,i]*exp(e[j,i]) / sum_k(S[k,i]*exp(e[k,i]))',
        'ncsm_prior': args.ncsm_prior,
        'ncsm_equation_status': 'include the explicit normalized NCSM prior in manuscript equations',
        'chi_affects_numeric_prior': args.ncsm_prior == 'distance',
        'ncsm_normalization': 'global maximum; self-distance = chi * min connected nonself distance',
        'degree_definition': args.degree_mode + '; binary adjacency; self-loops excluded',
        'input_scaling': 'per channel; originally observed raw training values only',
        'preprocessing': preprocessing,
        'discarded_windows': [n-len(part) for n, part in zip(candidate_sizes, origins)],
        'timestamps_verified': timestamps is not None,
        'monthly_reference': args.month_reference,
        'decoder': 'repeated per-node graph context; zero initial states; future-time LSTM; direct linear output',
        'feature_count': 3, 'flow_index': args.flow_index,
        'sampling_assumption': ('compressed artificial cycles; not real traffic timestamps' if args.synthetic_smoke
                                else 'continuous five-minute rows; verify against original timestamps'),
        'evaluation': 'raw flow; MAE/RMSE include zeros; MAPE excludes abs(actual)<=threshold',
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
    }
    (output/'run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Device: {device}; split sizes: {metadata["split_sizes"]}; horizon: {spec.horizon}', flush=True)
    print(f'Window lengths: {spec.window_lengths}; monthly reference: {args.month_reference}', flush=True)
    print(f'NCSM prior: {args.ncsm_prior}; normalized prior-weighted attention', flush=True)
    best, stale, history = float('inf'), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        train_loss = run_epoch(model, loaders[0], graph, device, optimizer)
        val_loss = run_epoch(model, loaders[1], graph, device)
        history.append({'epoch': epoch, 'train_mse_scaled': train_loss, 'val_mse_scaled': val_loss})
        print(f'Epoch {epoch}: train MSE={train_loss:.6f}; val MSE={val_loss:.6f}', flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            state = dict(metadata, epoch=epoch, validation_mse=best, model_state=model.state_dict(),
                         optimizer_state=optimizer.state_dict(),
                         graph={'degrees': graph[0].cpu(), 'edge_index': graph[1].cpu(),
                                'edge_weight': graph[2].cpu()})
            temp = checkpoint_path.with_suffix('.tmp')
            torch.save(state, temp)
            temp.replace(checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    with (output/'loss_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get('implementation_version') != 3:
        raise ValueError('This checkpoint predates the always-weighted NCSM implementation; retrain explicitly.')
    model = MSDSTLG(Config(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    graph = tuple(checkpoint['graph'][key].to(device) for key in ('degrees', 'edge_index', 'edge_weight'))
    restored_output_scaler = FeatureScaler.from_state_dict(checkpoint['output_scaler'])
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in loaders[2]:
            predictions.append(model(move_inputs(inputs, device), *graph).cpu().numpy())
    predictions = restored_output_scaler.inverse_transform(np.concatenate(predictions))
    label_indices = origins[2][:, None] + np.arange(1, spec.horizon+1)
    actuals = raw_data[label_indices, :, args.flow_index:args.flow_index+1]
    results = forecast_metrics(actuals, predictions, args.mape_threshold)
    results.update(best_epoch=checkpoint['epoch'], elapsed_seconds=time.perf_counter()-started,
                   synthetic_smoke=args.synthetic_smoke)
    (output/'metrics.json').write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding='utf-8')
    with (output/'metrics_by_horizon.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results['per_horizon'][0]))
        writer.writeheader()
        writer.writerows(results['per_horizon'])
    np.savez_compressed(output/'test_predictions.npz', predictions=predictions, actuals=actuals,
                        origins=origins[2], label_indices=label_indices)
    print('Test metrics (pooled):', results['pooled'], flush=True)
    if args.synthetic_smoke:
        print('Synthetic engineering test only; these are NOT PeMS benchmark results.', flush=True)
    return results


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(root/'data/PEMS08/PEMS08.npz'))
    parser.add_argument('--adjacency', default=str(root/'data/PEMS08/adj_PEMS08.pkl'))
    parser.add_argument('--distance', default=str(root/'data/PEMS08/adj_PEMS08_distance.pkl'))
    parser.add_argument('--timestamps', help='Optional .npy datetime/string vector; otherwise use NPZ timestamps.')
    parser.add_argument('--output-dir', default=str(root/'runs'/time.strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('--alpha', type=int, default=6)
    parser.add_argument('--beta', type=int, default=2)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--xi', type=float, default=1.0)
    parser.add_argument('--chi', type=float, default=0.1)
    parser.add_argument('--ncsm-prior', choices=('distance', 'binary'), default='distance',
                        help='distance: full NCSM model; binary: all-one prior for a controlled ablation.')
    parser.add_argument('--month-days', type=int, default=30)
    parser.add_argument('--month-reference', choices=('fixed_days', 'calendar_month'), default='fixed_days')
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--flow-index', type=int, default=0)
    parser.add_argument('--degree-mode', choices=('neighbors', 'in', 'out'), default='neighbors')
    parser.add_argument('--max-interpolation-gap', type=int, default=1,
                        help='Fill bounded NaN gaps up to this many samples; 1 is an implementation default.')
    parser.add_argument('--missing-value', type=float,
                        help='Explicit missing sentinel across input channels; zeros are observed by default.')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=26)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--degree-dim', type=int, default=16)
    parser.add_argument('--out-channels', type=int, default=56)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--decoder-hidden', type=int, default=20)
    parser.add_argument('--encoder-layers', type=int, default=1)
    parser.add_argument('--decoder-layers', type=int, default=1)
    parser.add_argument('--mape-threshold', type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    train_and_evaluate(parse_args())
