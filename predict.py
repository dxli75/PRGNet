import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool

class ParallelGNNModel(nn.Module):
    def __init__(self, node_dim=74, edge_dim=10, global_dim=8, hidden_dim=128, dropout=0.1, heads=1, num_gcn_layers=3, num_gat_layers=1):
        super().__init__()
        self.num_gcn_layers = num_gcn_layers
        self.num_gat_layers = num_gat_layers
        self.initial_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        self.gcn_dropouts = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn_bns.append(nn.BatchNorm1d(hidden_dim))
            self.gcn_dropouts.append(nn.Dropout(dropout))
        self.res_proj_gcn = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])
        self.gat_layers = nn.ModuleList()
        self.gat_bns = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()
        self.res_proj_gat = nn.ModuleList()
        gat_input_dim = hidden_dim
        for i in range(num_gat_layers):
            if i < num_gat_layers - 1:
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels=gat_input_dim,
                        out_channels=hidden_dim,
                        heads=heads,
                        edge_dim=edge_dim
                    )
                )
                out_dim = hidden_dim * heads
                self.gat_bns.append(nn.BatchNorm1d(out_dim))
                self.gat_dropouts.append(nn.Dropout(dropout))
                self.res_proj_gat.append(nn.Linear(gat_input_dim, out_dim))
                gat_input_dim = out_dim
            else:
                self.gat_layers.append(
                    GATv2Conv(
                        in_channels=gat_input_dim,
                        out_channels=hidden_dim,
                        heads=1,
                        edge_dim=edge_dim
                    )
                )
                out_dim = hidden_dim
                self.gat_bns.append(nn.BatchNorm1d(out_dim))
                self.gat_dropouts.append(nn.Dropout(dropout))
                self.res_proj_gat.append(nn.Linear(gat_input_dim, out_dim))
                gat_input_dim = out_dim
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    def forward(self, data):
        x = self.initial_embed(data.x)
        x_gcn = x.clone()
        for i in range(self.num_gcn_layers):
            x_gcn_init = x_gcn.clone()
            x_gcn = self.gcn_layers[i](x_gcn, data.edge_index)
            x_gcn = self.gcn_bns[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = self.gcn_dropouts[i](x_gcn)
            residual = self.res_proj_gcn[i](x_gcn_init)
            x_gcn = x_gcn + residual
        x_gat = x.clone()
        for i in range(self.num_gat_layers):
            x_gat_init = x_gat.clone()
            x_gat = self.gat_layers[i](x_gat, data.edge_index, data.edge_attr)
            x_gat = self.gat_bns[i](x_gat)
            x_gat = F.relu(x_gat)
            x_gat = self.gat_dropouts[i](x_gat)
            residual = self.res_proj_gat[i](x_gat_init)
            x_gat = x_gat + residual
        gat_pool = global_add_pool(x_gat, batch=data.batch)
        gcn_pool = global_add_pool(x_gcn, batch=data.batch)
        combined_pool = torch.cat([gcn_pool, gat_pool], dim=-1)
        fused_pool = self.feature_fusion(combined_pool)
        global_feat = self.global_mlp(data.global_features.view(data.global_features.size(0), -1))
        fused_feat = torch.cat([fused_pool, global_feat], dim=-1)
        return self.fc(fused_feat).view(-1)

def load_atom_graphs(data_path):
    data_list = []
    raw_data = np.load(data_path, allow_pickle=True)
    for idx, item in enumerate(raw_data):
        try:
            x = torch.tensor(item['x'], dtype=torch.float)
            edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(item['edge_attr'], dtype=torch.float)
            global_features = torch.tensor(item['global_features'], dtype=torch.float)
            global_features = global_features.view(1, -1)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_features=global_features
            )
            if 'metadata' in item and 'id' in item['metadata']:
                data.id = item['metadata']['id']
            data_list.append(data)
        except Exception as e:
            print(f"跳过图{idx}: {str(e)}")
    data_list = [d for d in data_list if d.edge_index.shape[1] > 0 and d.x.shape[0] > 0]
    print(f"成功加载有效图数量: {len(data_list)}/{len(raw_data)}")
    return data_list

def predict_unlabeled(data_path, model_path, batch_size=32, device='cuda:0', save_path='pred_results.csv'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = load_atom_graphs(data_path)
    loader = DataLoader(dataset, batch_size=batch_size)
    model = ParallelGNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    preds = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            if hasattr(batch, 'id'):
                ids.extend(batch.id)
            else:
                ids.extend([None]*batch.num_graphs)
    import pandas as pd
    df = pd.DataFrame({
        'index': range(len(preds)),
        'id': ids,
        'prediction': preds
    })
    df.to_csv(save_path, index=False)
    print(f'预测结果已保存到: {save_path}')

if __name__ == '__main__':
    DATA_PATH = './dataset/atom_feature/atom_graphs_unlabeled.npy'
    MODEL_PATH = './model/model_PRGNet.pth'
    SAVE_PATH = './pred_results.csv'
    predict_unlabeled(DATA_PATH, MODEL_PATH, batch_size=32, device='cuda:0', save_path=SAVE_PATH) 
