# utils.py
import os
import numpy as np
import torch
import resource
import multiprocessing
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index


def setup_device(gpu_id=0):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.backends.cudnn.benchmark = True
        try:
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        except Exception:
            print(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_atom_graphs(data_path, verbose=True):
    raw = np.load(data_path, allow_pickle=True)
    data_list = []

    for idx, item in enumerate(list(raw)):
        try:
            if isinstance(item, dict) and 'graph' in item and isinstance(item['graph'], Data):
                g = item['graph']
                g.complex_id = item.get('complex_id', f"complex_{idx}")
                if hasattr(g, 'global_features') and g.global_features is not None:
                    gf = torch.tensor(g.global_features, dtype=torch.float)
                    g.global_features = gf.view(1, -1) if gf.dim() == 1 else gf
                data_list.append(g)
                continue

            if isinstance(item, Data):
                g = item
                g.complex_id = getattr(g, "complex_id", f"complex_{idx}")
                if hasattr(g, 'global_features') and g.global_features is not None:
                    gf = torch.tensor(g.global_features, dtype=torch.float)
                    g.global_features = gf.view(1, -1) if gf.dim() == 1 else gf
                data_list.append(g)
                continue

            if isinstance(item, dict):
                x = torch.tensor(item['x'], dtype=torch.float)
                edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
                edge_attr = torch.tensor(item['edge_attr'], dtype=torch.float) if item.get('edge_attr') is not None else None
                y = torch.tensor(item.get('y', item.get('label')), dtype=torch.float).view(-1)
                gf = torch.tensor(item.get('global_features', np.zeros((1, 8))), dtype=torch.float)
                gf = gf.view(1, -1) if gf.dim() == 1 else gf
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, global_features=gf)
                data.complex_id = item.get("complex_id", f"complex_{idx}")
                data_list.append(data)
        except Exception:
            continue

    return [d for d in data_list if d.x is not None and d.edge_index is not None]


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {k: np.nan for k in ['MSE','RMSE','MAE','R2','Pearson r','CI','SD']}

    mse = mean_squared_error(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Pearson r': pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan,
        'CI': concordance_index(y_true, y_pred) if len(y_true) > 1 else np.nan,
        'SD': np.std(y_true - y_pred)
    }
