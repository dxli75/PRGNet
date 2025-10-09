import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool, global_mean_pool, global_max_pool, SAGPooling
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from tqdm import tqdm
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# ========== 设备配置 ==========
def setup_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
# ========== 数据加载与预处理 ==========
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
            y = torch.tensor(item['y'], dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                global_features=global_features
            )
            data_list.append(data)
        except Exception as e:
            print(f"跳过图{idx}: {str(e)}")
    
    data_list = [d for d in data_list if d.edge_index.shape[1] > 0 and d.x.shape[0] > 0]
    print(f"成功加载有效图数量: {len(data_list)}/{len(raw_data)}")
    return data_list
# ========== 新增交叉注意力模块 ==========
class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim, global_dim, hidden_dim, num_heads, dropout):
        super().__init__()

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_feat, global_feat):
        Q = self.graph_proj(graph_feat).unsqueeze(1)  # (B, 1, hidden)
        K = self.global_proj(global_feat).unsqueeze(1) # (B, 1, hidden)
        V = K

        attn_output, _ = self.attention(Q, K, V)
        attn_output = self.dropout(attn_output.squeeze(1))

        graph_proj = Q.squeeze(1)
        combined = torch.cat([graph_proj, attn_output], dim=-1)
        gate = self.gate(combined)
        fused = gate * graph_proj + (1 - gate) * attn_output
        
        fused = self.norm1(fused + graph_proj)
        
        ff_output = self.ffn(fused)
        
        output = self.norm2(fused + ff_output)
        return output

# ========== 模型定义 ==========
class ParallelGNNModel(nn.Module):
    def __init__(self, node_dim=89, edge_dim=24, global_dim=8, hidden_dim=128, dropout=0.1, heads=1, num_gcn_layers=3, num_gat_layers=1):
        super().__init__()
        self.num_gcn_layers = num_gcn_layers
        self.num_gat_layers = num_gat_layers
        
        self.initial_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== GCN =====
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
        
        # ===== GATv2 =====
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

#        self.cross_attn = CrossAttentionFusion(
#            graph_dim=hidden_dim,
#            global_dim=64,
#            hidden_dim=hidden_dim,  
#            num_heads=heads,
#            dropout=dropout
#        )

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
        
        # ===== GCN =====
        x_gcn = x.clone()
        for i in range(self.num_gcn_layers):
            x_gcn_init = x_gcn.clone()
            x_gcn = self.gcn_layers[i](x_gcn, data.edge_index)
            x_gcn = self.gcn_bns[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = self.gcn_dropouts[i](x_gcn)
            residual = self.res_proj_gcn[i](x_gcn_init)
            x_gcn = x_gcn + residual

        # ===== GATv2 =====
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

# ========== 评价指标计算 ==========
def compute_metrics(y_true, y_pred):
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0 or len(y_pred) == 0:
        print("警告：全部为无效值，返回默认指标")
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan,
                'R2m': np.nan, 'Pearson r': np.nan, 'CI': np.nan, 'SD': np.nan}
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson_r = pearsonr(y_true, y_pred)[0]
        ci = concordance_index(y_true, y_pred)
        residuals = y_true - y_pred
        sd = np.std(residuals)
        r2m = r2 * (1 - np.sqrt(abs(r2 - pearson_r**2)))
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'R2m': r2m, 'Pearson r': pearson_r, 'CI': ci, 'SD': sd}
    except Exception as e:
        print(f'计算指标时出错: {e}')
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan,
                'R2m': np.nan, 'Pearson r': np.nan, 'CI': np.nan, 'SD': np.nan}
            
# ========== 训练流程 ==========
def train_model(data_path, batch_size, epochs, lr, gpu_id, k_folds, num_gcn_layers, num_gat_layers):
    device = setup_device(gpu_id)
    dataset = load_atom_graphs(data_path)

    os.makedirs('./model_final_pli', exist_ok=True)
    os.makedirs('./training_plots', exist_ok=True)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    models = []
    fold_history = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n=== 开始训练 第{fold+1}/{k_folds}折 ===")
        train_dataset = [dataset[i] for i in train_val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.1, random_state=42)
        val_dataset = [train_dataset[i] for i in val_idx]
        train_dataset = [train_dataset[i] for i in train_idx]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
        model_save_name = f"model-3_fold{fold+1}.pth"
        model_save_path = os.path.join('./model_final_pli', model_save_name)
        print(f"模型保存路径: {model_save_path}")

        model = ParallelGNNModel(
            dropout=dropout,
            heads=heads,
            num_gcn_layers=num_gcn_layers,
            num_gat_layers=num_gat_layers
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scaler = GradScaler()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True,
            min_lr=1e-7
        )

        early_stop_cfg = { "patience": 60, "delta": 0.001, "metric": "RMSE", "verbose": True }
        best_val_metric = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        fold_metrics = {'train': [], 'val': []}

        start_time = time.time()
        epoch_times = []
        print("\n=== 开始训练 ===")
        for epoch in range(epochs):
            epoch_start = time.time()

            model.train()
            train_preds, train_labels = [], []
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                with autocast():
                    pred = model(batch)
                    loss = criterion(pred, batch.y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * batch.num_graphs
                train_preds.append(pred.detach().cpu().numpy())
                train_labels.append(batch.y.cpu().numpy())

            train_loss /= len(train_dataset)
            train_preds = np.concatenate(train_preds)
            train_labels = np.concatenate(train_labels)
            train_metrics = compute_metrics(train_labels, train_preds)
        
            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    loss = criterion(pred, batch.y)
                    val_loss += loss.item() * batch.num_graphs
                    val_preds.append(pred.cpu().numpy())
                    val_labels.append(batch.y.cpu().numpy())
        
            val_loss /= len(val_dataset)
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            val_metrics = compute_metrics(val_labels, val_preds)
        
            fold_metrics['train'].append(train_metrics)
            fold_metrics['val'].append(val_metrics)
            scheduler.step(val_metrics[early_stop_cfg["metric"]])

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            average_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = average_time * remaining_epochs

            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds = int(seconds % 60)
                return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

            print(f"\nEpoch {epoch+1}/{epochs}[{epoch_time:.1f}s]")
            print(f"累计时间: {format_time(time.time() - start_time)}")
            print(f"预估剩余: {format_time(remaining_time)}")
            print(f"[Train] Loss: {train_loss:.4f}, " + ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
            print(f"[Val]   Loss: {val_loss:.4f}, " + ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
   
            current_val_metric = val_metrics[early_stop_cfg["metric"]]
            if current_val_metric < (best_val_metric + early_stop_cfg["delta"]):
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), model_save_path)
                if early_stop_cfg["verbose"]:
                    print(f"验证指标提升: {best_val_metric:.4f}，保存模型到 {model_save_path}")
            else:
                epochs_no_improve += 1
                if early_stop_cfg["verbose"]:
                    remaining = early_stop_cfg["patience"] - epochs_no_improve
                    print(f"早停计数器: {epochs_no_improve}/{early_stop_cfg['patience']}(剩余耐心: {remaining})")

            if epochs_no_improve >= early_stop_cfg["patience"]:
                print(f"\n早停触发! 在epoch {epoch+1}停止训练")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        fold_history.append(fold_metrics)

        print("\n=== 开始最终测试 ===")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        models.append(model)
        
        model.eval()
        test_preds, test_labels = [], []
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                test_loss += loss.item() * batch.num_graphs
                test_preds.append(pred.cpu().numpy())
                test_labels.append(batch.y.cpu().numpy())
    
        test_loss /= len(test_dataset)
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        test_metrics = compute_metrics(test_labels, test_preds)
        fold_results.append(test_metrics)

        print(f"\n=== 第{fold+1}折测试结果 ===")
        print(f"Test Loss: {test_loss:.4f}")
        print(", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))

    avg_metrics = {}
    std_metrics = {}
    for key in fold_results[0].keys():
        values = [m[key] for m in fold_results]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    print(f"\n=== 平均测试结果 ===")
    print(", ".join([f"{k}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}" for k in avg_metrics.keys()]))

    return avg_metrics, std_metrics


if __name__ == "__main__":
    DATA_PATH = "./dataset/atom_feature/atom_graphs.npy"
    gpu_id = 0
    k_folds = 5
    epochs = 1000
    base_lr = 0.001
    base_bs = 64
    batch_size = 32
    lr = base_lr * batch_size / base_bs
    dropout = 0.1
    heads = 1
    num_gcn_layers = 3
    num_gat_layers = 1

    results = train_model(
        data_path=DATA_PATH,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        gpu_id=gpu_id,
        k_folds=k_folds,
        num_gcn_layers=num_gcn_layers,
        num_gat_layers=num_gat_layers
    )
