# PRGNet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool


class ParallelGNNModel(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        global_dim,
        hidden_dim=128,
        dropout=0.1,
        heads=1,
        num_gcn_layers=3,
        num_gat_layers=1
    ):
        super().__init__()

        self.num_gcn_layers = num_gcn_layers
        self.num_gat_layers = num_gat_layers

        self.initial_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== GCN branch =====
        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        self.gcn_dropouts = nn.ModuleList()
        self.res_proj_gcn = nn.ModuleList()

        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn_bns.append(nn.BatchNorm1d(hidden_dim))
            self.gcn_dropouts.append(nn.Dropout(dropout))
            self.res_proj_gcn.append(nn.Linear(hidden_dim, hidden_dim))

        # ===== GATv2 branch =====
        self.gat_layers = nn.ModuleList()
        self.gat_bns = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()
        self.res_proj_gat = nn.ModuleList()

        gat_input_dim = hidden_dim
        for i in range(num_gat_layers):
            heads_i = heads if i < num_gat_layers - 1 else 1
            out_channels = hidden_dim
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=gat_input_dim,
                    out_channels=out_channels,
                    heads=heads_i,
                    edge_dim=edge_dim
                )
            )
            out_dim = out_channels * (heads_i if i < num_gat_layers - 1 else 1)
            self.gat_bns.append(nn.BatchNorm1d(out_dim))
            self.gat_dropouts.append(nn.Dropout(dropout))
            self.res_proj_gat.append(nn.Linear(gat_input_dim, out_dim))
            gat_input_dim = out_dim

        # ===== Global feature MLP =====
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

        # ===== Feature fusion =====
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

        # ===== Regression head =====
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

        # ----- GCN -----
        x_gcn = x
        for i in range(self.num_gcn_layers):
            x_init = x_gcn
            x_gcn = self.gcn_layers[i](x_gcn, data.edge_index)
            x_gcn = self.gcn_bns[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = self.gcn_dropouts[i](x_gcn)
            x_gcn = x_gcn + self.res_proj_gcn[i](x_init)

        # ----- GAT -----
        x_gat = x
        for i in range(self.num_gat_layers):
            x_init = x_gat
            x_gat = self.gat_layers[i](x_gat, data.edge_index, data.edge_attr)
            x_gat = self.gat_bns[i](x_gat)
            x_gat = F.relu(x_gat)
            x_gat = self.gat_dropouts[i](x_gat)
            residual = self.res_proj_gat[i](x_init)
            if residual.shape != x_gat.shape:
                residual = residual[:, :x_gat.shape[1]]
            x_gat = x_gat + residual

        gcn_pool = global_add_pool(x_gcn, data.batch)
        gat_pool = global_add_pool(x_gat, data.batch)

        fused_pool = self.feature_fusion(torch.cat([gcn_pool, gat_pool], dim=-1))
        global_feat = self.global_mlp(data.global_features.view(data.global_features.size(0), -1))
        fused_feat = torch.cat([fused_pool, global_feat], dim=-1)

        return self.fc(fused_feat).view(-1)
