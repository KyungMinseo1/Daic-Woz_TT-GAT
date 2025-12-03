import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool

class GATClassifier(nn.Module):
    def __init__(self, text_dim, vision_dim, audio_dim, hidden_channels, num_classes, heads=8, dropout=0.1):
        """
        Args:
            text_dim: 텍스트 임베딩 차원 (summary, topic, transcription)
            vision_dim: vision 피처 원본 차원
            audio_dim: audio 피처 원본 차원
            hidden_channels: GAT hidden 차원
            num_classes: 분류 클래스 수
            heads: attention head 수
            dropout: dropout 비율
        """
        super().__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim

        self.text_proj = nn.Linear(text_dim, hidden_channels)
        self.vision_proj = nn.Linear(vision_dim, hidden_channels)
        self.audio_proj = nn.Linear(audio_dim, hidden_channels)

        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, add_self_loops=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout, add_self_loops=True)

        out_dim = 1 if num_classes == 2 else num_classes
        self.lin = nn.Linear(hidden_channels, out_dim)

    def project_features(self, x, node_types):
        """        
        Args:
            x: [num_nodes, padded_dim] tensor
            node_types: type of nodes
        
        Returns:
            [num_nodes, hidden_channels] tensor
        """
        device = x.device
        num_nodes = x.size(0)
        
        # generate masking
        text_mask = torch.tensor([nt in ['summary', 'topic', 'transcription'] 
                                  for nt in node_types], device=device)
        vision_mask = torch.tensor([nt == 'vision' for nt in node_types], device=device)
        audio_mask = torch.tensor([nt == 'audio' for nt in node_types], device=device)
        
        transformed = torch.zeros(num_nodes, self.text_proj.out_features, device=device)
        
        # Projection
        if text_mask.any():
            text_indices = text_mask.nonzero(as_tuple=True)[0]
            transformed[text_indices] = self.text_proj(x[text_indices, :self.text_dim])
        
        if vision_mask.any():
            vision_indices = vision_mask.nonzero(as_tuple=True)[0]
            transformed[vision_indices] = self.vision_proj(x[vision_indices, :self.vision_dim])
        
        if audio_mask.any():
            audio_indices = audio_mask.nonzero(as_tuple=True)[0]
            transformed[audio_indices] = self.audio_proj(x[audio_indices, :self.audio_dim])
        
        return transformed

    def forward(self, data):
        """
        Args:
            data: PyG Data 객체
                - data.x: 노드 피처
                - data.edge_index: 엣지 인덱스
                - data.batch: 배치 정보
                - data.node_types: 노드 타입 리스트 (len = num_nodes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_types = data.node_types

        # 선형 변환
        x = self.project_features(x, node_types)

        # GAT 레이어 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # GAT 레이어 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # batch 벡터를 이용해 그래프별 평균 계산
        graph_embeddings = global_mean_pool(x, batch) # [batch_size, hidden_channels]
        
        x = F.dropout(graph_embeddings, p=self.dropout, training=self.training)
        out = self.lin(x)
        if self.num_classes == 2:
            out = out.squeeze(-1)
        return out
    