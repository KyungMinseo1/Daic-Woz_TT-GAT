import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool, global_add_pool
from torch_geometric.nn.models import JumpingKnowledge
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
from loguru import logger

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

class GATClassifier(nn.Module):
    def __init__(
            self,
            text_dim,
            vision_dim,
            audio_dim,
            hidden_channels,
            num_layers,
            num_classes,
            dropout_dict,
            heads=8,
            use_summary_node=True,
            use_text_proj=True):
        """
        Args:
            text_dim: 텍스트 임베딩 차원 (summary, transcription)
            vision_dim: vision 피처 원본 차원
            audio_dim: audio 피처 원본 차원
            hidden_channels: GAT hidden 차원
            num_layers: GAT 레이어 수
            num_classes: 분류 클래스 수
            heads: attention head 수
            dropout: dropout 비율
            use_summary_node: Summary Node 사용 여부
            use_text_proj: Transcription Projection layer 사용 여부
        """
        super().__init__()
        
        assert 2 <= num_layers and num_layers <= 4, logger.error("Number of Layers should be set between 2 and 4")
        
        self.dropout_t = dropout_dict.get('text_dropout', 0.1)
        self.dropout_g = dropout_dict.get('graph_dropout', 0.1)
        self.dropout_v = dropout_dict.get('vision_dropout', 0.1)
        self.dropout_a = dropout_dict.get('audio_dropout', 0.1)
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_summary_node = use_summary_node
        self.use_text_proj = use_text_proj

        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        self.norm3 = nn.LayerNorm(hidden_channels * heads)
        self.norm4 = nn.LayerNorm(hidden_channels)

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_v)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_a)
        )

        # Text & Proxy & Topic & Summary Projection
        if self.use_text_proj:
            self.text_proj = nn.Linear(text_dim, hidden_channels)
        self.dropout_text = nn.Dropout(self.dropout_t)

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv4 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout_g, add_self_loops=True)

        out_dim = 1 if num_classes == 2 else num_classes
        self.lin = nn.Linear(hidden_channels, out_dim)

        self._init_weights()

    def _init_weights(self):
        """GAT 초기화"""
        
        # Linear
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier -> ELU/Relu
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm -> weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # GATv2Conv
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        
        nn.init.xavier_uniform_(self.lin.weight, gain=0.01)

    def forward(self, data, explanation=False):
        """
        Args:
            data: PyG Data 객체
                - data.x: 노드 피처
                - data.edge_index: 엣지 인덱스
                - data.batch: 배치 정보
                - data.node_types: 노드 타입 리스트 (len = num_nodes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_vision = data.x_vision    # (N_vision, seq_len, v_dim)
        x_audio = data.x_audio      # (N_audio, seq_len, a_dim)
        node_types = data.node_types

        if self.use_text_proj:
            x_proj = self.text_proj(x)
        else:
            x_proj = x

        text_features = self.dropout_text(x_proj)
        final_x = text_features.clone()

        flat_node_types = []
        if isinstance(node_types[0], list):
            for sublist in node_types: flat_node_types.extend(sublist)
        else:
            flat_node_types = node_types
            
        vision_indices = [i for i, t in enumerate(flat_node_types) if t == 'vision']
        audio_indices = [i for i, t in enumerate(flat_node_types) if t == 'audio']

        # Vision
        if x_vision.size(0) > 0 and len(vision_indices) > 0:
            h_vision = self.vision_proj(x_vision) # (N_vis, Hidden)
            if len(vision_indices) == h_vision.size(0):
                mask_weight = x[vision_indices].mean(dim=1, keepdim=True)
                weighted_vision = h_vision.to(final_x.dtype) * mask_weight.to(final_x.dtype)
                final_x[vision_indices] = weighted_vision

        # Audio
        if x_audio.size(0) > 0 and len(audio_indices) > 0:
            h_audio = self.audio_proj(x_audio) # (N_aud, Hidden)
            if len(audio_indices) == h_audio.size(0):
                mask_weight = x[audio_indices].mean(dim=1, keepdim=True)
                weighted_audio = h_audio.to(final_x.dtype) * mask_weight.to(final_x.dtype)
                final_x[audio_indices] = weighted_audio
        
        # GAT 레이어 1
        x = F.dropout(final_x, p=self.dropout_g, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
		        
        if self.num_layers >= 3:
            x_in = x
            x = F.dropout(x, p=self.dropout_g, training=self.training)
            x = self.conv2(x, edge_index)
            x = self.norm2(x + x_in)
            x = F.elu(x)
          
        if self.num_layers >= 4:
            x_in = x
            x = F.dropout(x, p=self.dropout_g, training=self.training)
            x = self.conv3(x, edge_index)
            x = self.norm3(x + x_in)
            x = F.elu(x)
        
        # GAT 레이어 4
        x = F.dropout(x, p=self.dropout_g, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        
        # batch 벡터를 이용해 그래프별 평균 계산
        # graph_embeddings = global_mean_pool(x, batch) # [batch_size, hidden_channels]

        if self.use_summary_node:
          # Summary node로 최종 로짓값 산출
          summary_nodes = x[data.ptr[:-1]]

          # Classification
          # x = F.dropout(graph_embeddings, p=self.dropout, training=self.training)
          out = F.dropout(summary_nodes, p=self.dropout_g, training=self.training)
          
        else:
          # Summary node가 할당되지 않았다고 가정
          # Topic node의 global pool 진행
          topic_indices = [i for i, t in enumerate(flat_node_types) if t == 'topic']
          topic_x = x[topic_indices]
          topic_batch = batch[topic_indices]
          if len(topic_x) > 0:
                topic_nodes = global_mean_pool(topic_x, topic_batch)
                out = F.dropout(topic_nodes, p=self.dropout_g, training=self.training)
          else:
                out = global_mean_pool(x, batch)
          
        out = self.lin(out)
        if self.num_classes == 2:
            out = out.squeeze(-1)
        if explanation:
            return out, x, flat_node_types
        else:
            return out
    
class GATJKClassifier(nn.Module):
    def __init__(
            self,
            text_dim,
            vision_dim,
            audio_dim,
            hidden_channels,
            num_layers,
            num_classes,
            dropout_dict,
            heads=8,
            use_summary_node=True,
            use_text_proj=True):
        """
        Args:
            text_dim: 텍스트 임베딩 차원 (summary, transcription)
            vision_dim: vision 피처 원본 차원
            audio_dim: audio 피처 원본 차원
            hidden_channels: GAT hidden 차원
            num_layers: GAT 레이어 수
            num_classes: 분류 클래스 수
            heads: attention head 수
            dropout: dropout 비율
            use_summary_node: Summary Node 사용 여부
            use_text_proj: Transcription Projection layer 사용 여부
        """
        super().__init__()
        
        assert 2 <= num_layers and num_layers <= 4, logger.error("Number of Layers should be set between 2 and 4")
        
        self.dropout_t = dropout_dict.get('text_dropout', 0.1)
        self.dropout_g = dropout_dict.get('graph_dropout', 0.1)
        self.dropout_v = dropout_dict.get('vision_dropout', 0.1)
        self.dropout_a = dropout_dict.get('audio_dropout', 0.1)
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_summary_node = use_summary_node
        self.use_text_proj = use_text_proj

        self.norm1 = GraphNorm(hidden_channels * heads)
        self.norm2 = GraphNorm(hidden_channels * heads)
        self.norm3 = GraphNorm(hidden_channels * heads)
        self.norm4 = GraphNorm(hidden_channels)

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_v)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_a)
        )

        # Text & Proxy & Topic & Summary Projection
        if self.use_text_proj:
            self.text_proj = nn.Linear(text_dim, hidden_channels)
        self.dropout_text = nn.Dropout(self.dropout_t)

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv4 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout_g, add_self_loops=True)

        self.jk = JumpingKnowledge(mode='cat')

        out_dim = 1 if num_classes == 2 else num_classes
        
         # JK 차원 계산
        if self.num_layers == 2:
            final_dim = (hidden_channels * heads) * 1 + hidden_channels
        elif self.num_layers == 3:
            final_dim = (hidden_channels * heads) * 2 + hidden_channels
        else:  # num_layers == 4
            final_dim = (hidden_channels * heads) * 3 + hidden_channels

        self.lin = nn.Linear(final_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        """GAT 초기화"""
        
        # Linear
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier -> ELU/Relu
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm -> weight=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # GATv2Conv
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        
        nn.init.xavier_uniform_(self.lin.weight, gain=0.01)

    def forward(self, data, explanation=False):
        """
        Args:
            data: PyG Data 객체
                - data.x: 노드 피처
                - data.edge_index: 엣지 인덱스
                - data.batch: 배치 정보
                - data.node_types: 노드 타입 리스트 (len = num_nodes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_vision = data.x_vision    # (N_vision, seq_len, v_dim)
        x_audio = data.x_audio      # (N_audio, seq_len, a_dim)
        node_types = data.node_types

        if self.use_text_proj:
            x_proj = self.text_proj(x)
        else:
            x_proj = x

        text_features = self.dropout_text(x_proj)
        final_x = text_features.clone()

        flat_node_types = []
        if isinstance(node_types[0], list):
            for sublist in node_types: flat_node_types.extend(sublist)
        else:
            flat_node_types = node_types
            
        vision_indices = [i for i, t in enumerate(flat_node_types) if t == 'vision']
        audio_indices = [i for i, t in enumerate(flat_node_types) if t == 'audio']

        # Vision
        if x_vision.size(0) > 0 and len(vision_indices) > 0:
            h_vision = self.vision_proj(x_vision) # (N_vis, Hidden)
            if len(vision_indices) == h_vision.size(0):
                mask_weight = x[vision_indices].mean(dim=1, keepdim=True)
                weighted_vision = h_vision.to(final_x.dtype) * mask_weight.to(final_x.dtype)
                final_x[vision_indices] = weighted_vision

        # Audio
        if x_audio.size(0) > 0 and len(audio_indices) > 0:
            h_audio = self.audio_proj(x_audio) # (N_aud, Hidden)
            if len(audio_indices) == h_audio.size(0):
                mask_weight = x[audio_indices].mean(dim=1, keepdim=True)
                weighted_audio = h_audio.to(final_x.dtype) * mask_weight.to(final_x.dtype)
                final_x[audio_indices] = weighted_audio
        
        xs = []

        # GAT 레이어 1
        x = F.dropout(final_x, p=self.dropout_g, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        xs.append(x)
		        
        if self.num_layers >= 3:
            x_in = x
            x = F.dropout(x, p=self.dropout_g, training=self.training)
            x = self.conv2(x, edge_index)
            x = self.norm2(x + x_in)
            x = F.elu(x)
            xs.append(x)
          
        if self.num_layers >= 4:
            x_in = x
            x = F.dropout(x, p=self.dropout_g, training=self.training)
            x = self.conv3(x, edge_index)
            x = self.norm3(x + x_in)
            x = F.elu(x)
            xs.append(x)
        
        # GAT 레이어 4
        x = F.dropout(x, p=self.dropout_g, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        xs.append(x)

        # Jumping Knowledge로 모든 레이어 정보 통합
        x = self.jk(xs) # [num_nodes, final_dim]
        
        # batch 벡터를 이용해 그래프별 평균 계산
        # graph_embeddings = global_mean_pool(x, batch) # [batch_size, hidden_channels]

        if self.use_summary_node:
          # Summary node로 최종 로짓값 산출
          summary_nodes = x[data.ptr[:-1]]

          # Classification
          # x = F.dropout(graph_embeddings, p=self.dropout, training=self.training)
          out = F.dropout(summary_nodes, p=self.dropout_g, training=self.training)
          
        else:
          # Summary node가 할당되지 않았다고 가정
          # Topic node의 global pool 진행
          topic_indices = [i for i, t in enumerate(flat_node_types) if t == 'topic']
          topic_x = x[topic_indices]
          topic_batch = batch[topic_indices]
          if len(topic_x) > 0:
                topic_nodes = global_mean_pool(topic_x, topic_batch)
                out = F.dropout(topic_nodes, p=self.dropout_g, training=self.training)
          else:
                out = global_mean_pool(x, batch)
                out = F.dropout(out, p=self.dropout_g, training=self.training)
          
        out = self.lin(out)
        if self.num_classes == 2:
            out = out.squeeze(-1)
        if explanation:
            return out, x, flat_node_types
        else:
            return out