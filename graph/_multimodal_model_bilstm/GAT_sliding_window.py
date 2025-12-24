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


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.lin2 = nn.Linear(hidden_dim//2, 1)
    def forward(self, x):
        scores = torch.tanh(self.lin1(x))
        scores = self.lin2(scores) # (N, timestamp, 1)
        attention_weights = F.softmax(scores, dim=1)
        weighted_output = x*attention_weights # (N, timestamp, hidden_dim)
        context_vector = torch.sum(weighted_output, dim=1)
        return context_vector

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.attention = Attention(hidden_dim * 2)
        # self.multihead_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self._init_weights()

    def _init_weights(self):
        """LSTM, Linear 초기화"""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                # Input-Hidden: Xavier Uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-Hidden: Orthogonal
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias: 0
                nn.init.zeros_(param.data)
                # param.data[self.bilstm.hidden_size:2*self.bilstm.hidden_size] = 1.0

        # Attention 초기화
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.fc.weight) 
        # nn.init.normal_(self.fc.weight, mean=0, std=0.01) 
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, lengths):
        """
        Args:
        x: (batch, seq_len, input_dim) - already feature vectors
        lengths: (batch,) - sequence lengths
        """
        self.bilstm.flatten_parameters()
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.bilstm(packed)

        # output shape: (B, seq_len, hidden*2)
        lstm_output, _ = pad_packed_sequence(packed_out, batch_first=True)

        context_vector = self.attention(lstm_output)
        # attn_output, attn_weights = self.multihead_attn(lstm_output, lstm_output, lstm_output)
        # context_vector = attn_output.mean(dim=1)

        # Attention
        out = self.dropout(context_vector)
        logits = self.fc(out)
        return logits, context_vector

class SimpleBiLSTM(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self._init_weights()

    def _init_weights(self):
        """LSTM, Linear 초기화"""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                # Input-Hidden: Xavier Uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-Hidden: Orthogonal
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias: 0
                nn.init.zeros_(param.data)
                # param.data[self.bilstm.hidden_size:2*self.bilstm.hidden_size] = 1.0

    def forward(self, x, lengths):
        """
        Args:
        x: (batch, seq_len, input_dim) - already feature vectors
        lengths: (batch,) - sequence lengths
        """
        self.bilstm.flatten_parameters()
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.bilstm(packed)

        if self.bilstm.bidirectional:
            # take the last layer's forward and backward
            # forward hidden is h_n[-2], backward hidden is h_n[-1]
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_final = torch.cat((h_forward, h_backward), dim=1) # (batch, hidden_dim*2)
        else:
            h_final = h_n[-1]

        out = self.dropout(h_final)
        logits = self.fc(out)
        return logits, h_final
    
   
class GATJKClassifier(nn.Module):
    def __init__(
            self,
            text_dim,
            vision_dim,
            audio_dim,
            hidden_channels,
            num_layers,
            bilstm_num_layers,
            num_classes,
            dropout_dict,
            heads=8,
            use_attention=False,
            use_summary_node=True,
            use_text_proj=True):
        """
        Args:
            text_dim: 텍스트 임베딩 차원 (summary, transcription)
            vision_dim: vision 피처 원본 차원
            audio_dim: audio 피처 원본 차원
            hidden_channels: GAT hidden 차원
            num_layers: GAT 레이어 수
            bilstm_num_layers: BiLSTM 레이어 수
            num_classes: 분류 클래스 수
            heads: attention head 수
            dropout: dropout 비율
            use_attention: AttentionLSTM 사용 여부
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
        self.use_attention = use_attention
        self.use_summary_node = use_summary_node
        self.use_text_proj = use_text_proj

        self.norm1 = GraphNorm(hidden_channels * heads)
        self.norm2 = GraphNorm(hidden_channels * heads)
        self.norm3 = GraphNorm(hidden_channels * heads)
        self.norm4 = GraphNorm(hidden_channels)

        # Vision LSTM
        if self.use_attention:
            self.vision_lstm = AttentionBiLSTM(
                input_dim=vision_dim, 
                hidden_dim=hidden_channels,
                output_dim=hidden_channels,
                num_layers=bilstm_num_layers,
                dropout=self.dropout_v
            )
        else:
            self.vision_lstm = SimpleBiLSTM(
                input_dim=vision_dim,
                hidden_dim=hidden_channels,
                output_dim=hidden_channels,
                num_layers=bilstm_num_layers,
                dropout=self.dropout_v
            )

        # Audio LSTM
        if self.use_attention:
            self.audio_lstm = AttentionBiLSTM(
                input_dim=audio_dim, 
                hidden_dim=hidden_channels,
                output_dim=hidden_channels,
                num_layers=bilstm_num_layers,
                dropout=self.dropout_a
            )
        else:
            self.audio_lstm = SimpleBiLSTM(
                input_dim=audio_dim,
                hidden_dim=hidden_channels,
                output_dim=hidden_channels,
                num_layers=bilstm_num_layers,
                dropout=self.dropout_a
            )
        
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

        if self.use_attention and hasattr(self.vision_lstm, 'bilstm'):
            self.vision_lstm.bilstm.flatten_parameters()
            self.audio_lstm.bilstm.flatten_parameters()
        elif hasattr(self.vision_lstm, 'bilstm'):
            self.vision_lstm.bilstm.flatten_parameters()
            self.audio_lstm.bilstm.flatten_parameters()

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
        x_vision = data.x_vision    # (Batch, Window_Num, Window_Size, v_dim)
        x_audio = data.x_audio      # (Batch, Window_Num, Window_Size, a_dim)
        node_types = data.node_types
        
        vision_lengths = data.vision_lengths    # (Batch, Window_Num)
        audio_lengths = data.audio_lengths      # (Batch, Window_Num)

        if self.use_text_proj:
            x = self.text_proj(x)

        text_features = self.dropout_text(x)
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
            h_win_v, _ = self.vision_lstm(x_vision, vision_lengths) # (Total_V_Windows, Hidden)
            h_vision = global_mean_pool(h_win_v, data.vision_node_map)
            if len(vision_indices) == h_vision.size(0):
                mask_weight = x[vision_indices].mean(dim=1, keepdim=True)
                weighted_vision = h_vision.to(final_x.dtype) * mask_weight.to(final_x.dtype)
                final_x[vision_indices] = weighted_vision

        # Audio
        if x_audio.size(0) > 0 and len(audio_indices) > 0:
            h_win_a, _ = self.audio_lstm(x_audio, audio_lengths) # (Total_A_Windows, Hidden)
            h_audio = global_mean_pool(h_win_a, data.audio_node_map)
            if len(audio_indices) == h_audio.size(0):
                mask_weight = x[audio_indices].mean(dim=1, keepdim=True)
                weighted_audio = h_audio.to(final_x.dtype) * mask_weight.to(final_x.dtype)
                final_x[audio_indices] = weighted_audio

        xs = []
        
        # GAT 레이어 1
        x = F.dropout(final_x, p=self.dropout_g, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.elu(x)
        xs.append(x)
		        
        if self.num_layers >= 3:
            # GAT 레이어 2
            x_in = x
            x = F.dropout(x, p=self.dropout_g, training=self.training)
            x = self.conv2(x, edge_index)
            x = self.norm2(x + x_in, batch)
            x = F.elu(x)
            xs.append(x)
          
        if self.num_layers >= 4:
            # GAT 레이어 3
            x_in = x
            x = F.dropout(x, p=self.dropout_g, training=self.training)
            x = self.conv3(x, edge_index)
            x = self.norm3(x + x_in, batch)
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
            if hasattr(data, 'ptr'):
                summary_nodes = x[data.ptr[:-1]]
            else:
                summary_nodes = x[0].unsqueeze(0)

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