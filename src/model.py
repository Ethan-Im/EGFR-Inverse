import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP

class EGFRAffinityModel(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=200,
                 out_channels=1, edge_dim=3, num_layers=3, num_timesteps=2, dropout=0.2):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.gnn(x, edge_index, edge_attr, batch)
