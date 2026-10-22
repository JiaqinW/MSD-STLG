# MSD-STLG: Code and Configuration


## Files and environment

- `main.py`: data loading, preprocessing, training, and evaluation.
- `model.py`: temporal encoders, node-degree encoding, NCSM-weighted GAT, and prediction decoder.
- `util.py`: temporal windows, missing-value processing, normalization, graph construction, and metrics.

```bash
python -m pip install numpy==2.5.4 torch==2.14.1
```

## Data and configuration

Provide an NPZ file containing `data` with shape `[time, nodes, 3]` for flow, speed, and occupancy. The flow channel defaults to index 0. Adjacency and physical-distance matrices must use the same node order as the traffic observations. Rows must represent consecutive five-minute intervals.

## Run

From the repository directory:

```bash
python main.py \
  --data data/PEMS08/PEMS08.npz \
  --adjacency data/PEMS08/adj_PEMS08.pkl \
  --distance data/PEMS08/adj_PEMS08_distance.pkl \
  --output-dir runs/PEMS08
```


