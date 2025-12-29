# train.py
import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.cuda.amp import GradScaler

from PRGNet_model import ParallelGNNModel
from utils import setup_device, load_atom_graphs, compute_metrics


def train_full_model(
    data_path,
    batch_size,
    epochs,
    lr,
    gpu_id,
    num_gcn_layers,
    num_gat_layers,
    dropout,
    heads,
    save_dir
):
    device = setup_device(gpu_id)

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log_full.csv")

    dataset = load_atom_graphs(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample = dataset[0]
    model = ParallelGNNModel(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1] if sample.edge_attr is not None else 0,
        global_dim=sample.global_features.shape[1],
        dropout=dropout,
        heads=heads,
        num_gcn_layers=num_gcn_layers,
        num_gat_layers=num_gat_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)

    best_rmse = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        preds, labels, loss_sum = [], [], 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                out = model(batch)
                loss = criterion(out, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * batch.num_graphs
            preds.append(out.detach().cpu().numpy())
            labels.append(batch.y.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        metrics = compute_metrics(labels, preds)
        scheduler.step(metrics['RMSE'])

        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))

        history.append(metrics)
        pd.DataFrame(history).to_csv(log_file, index=False)

    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return best_rmse


if __name__ == "__main__":
    DATA_PATH = "../graph/v2020/atom_graphs.npy"
    best_rmse = train_full_model(
        data_path=DATA_PATH,
        batch_size=32,
        epochs=1000,
        lr=0.001 * 32 / 64,
        gpu_id=1,
        num_gcn_layers=3,
        num_gat_layers=1,
        dropout=0.1,
        heads=1,
        save_dir="./model_v2020"
    )
    print("Best RMSE:", best_rmse)

