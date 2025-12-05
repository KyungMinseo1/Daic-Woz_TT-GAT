import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=8, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        
        # GAT Layer 1
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout, add_self_loops=True)
        # GAT Layer 2
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout, add_self_loops=True)
        
        out_dim = 1 if num_classes == 2 else num_classes
        self.lin = torch.nn.Linear(hidden_channels, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # GAT Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # [수정된 부분] Pooling Strategy
        # Option A: 모든 노드 평균 (기존 방식) -> global_mean_pool(x, batch)
        # Option B: Summary Node(0번)만 사용 (계층적 구조 활용) -> 권장!
        
        # data.ptr은 배치의 각 그래프 시작 인덱스를 담고 있음 [0, num_nodes_g1, num_nodes_g1+g2, ...]
        # ptr[:-1]을 하면 각 그래프의 첫 번째 노드(Summary Node)의 인덱스가 됨
        summary_nodes = x[data.ptr[:-1]] 
        
        # 최종 분류
        out = F.dropout(summary_nodes, p=0.5, training=self.training) # Dropout 추가 권장
        out = self.lin(out)
        
        if self.num_classes == 2:
            out = out.squeeze(-1)
            
        return out