import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# 1. GAT 모델 정의
class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=8, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        # GAT Layer 1: Multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout, add_self_loops=True)
        # GAT Layer 2: 최종 임베딩
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout, add_self_loops=True)
        # 분류를 위한 fully connected layer
        out_dim = 1 if num_classes == 2 else num_classes
        self.lin = torch.nn.Linear(hidden_channels, out_dim)

    def forward(self, data):
        # x: 노드 특징 [num_nodes, in_channels]
        # edge_index: 엣지 정보 [2, num_edges]
        # batch: 각 노드가 어느 그래프에 속하는지 [num_nodes]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # GAT 레이어 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # GAT 레이어 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 그래프별로 노드 임베딩을 평균내서 그래프 임베딩 생성
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        
        # 최종 분류
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)
        if self.num_classes == 2:
            out = out.squeeze(-1)
        return out