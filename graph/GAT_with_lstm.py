import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_feat, modal_feats):
        """
        Args:
            text_feat: (N_text, dim) - text node features
            modal_feats: (N_modal, dim) - vision + audio features
        """
        if modal_feats.size(0) == 0:
            return text_feat
        
        # Reshape for MultiheadAttention
        query = text_feat.unsqueeze(0)  # (1, N_text, dim)
        key_value = modal_feats.unsqueeze(0)  # (1, N_modal, dim)
        
        attn_output, _ = self.multihead_attn(
            query, 
            key_value, 
            key_value
        )  # (1, N_text, dim)
        
        attn_output = attn_output.squeeze(0) # (N_text, dim)
        
        # Residual connection + Normalization
        enhanced_text = self.norm(text_feat + self.dropout(attn_output))
        
        return enhanced_text

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        # self.attention = Attention(hidden_dim * 2)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

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

        # context_vector = self.attention(lstm_output)
        attn_output, attn_weights = self.multihead_attn(lstm_output, lstm_output, lstm_output)
        context_vector = attn_output.mean(dim=1)

        # Attention
        out = self.dropout(context_vector)
        logits = self.fc(out)
        return logits, context_vector

class GATClassifier(nn.Module):
    def __init__(self, text_dim, vision_dim, audio_dim, hidden_channels, num_classes, dropout_dict, heads=8, use_cross_modal=False):
        """
        Args:
            text_dim: 텍스트 임베딩 차원 (summary, topic, transcription)
            vision_dim: vision 피처 원본 차원
            audio_dim: audio 피처 원본 차원
            hidden_channels: GAT hidden 차원
            num_classes: 분류 클래스 수
            heads: attention head 수
            dropout: dropout 비율
            use_cross_modal: CrossModalAttention 사용 여부
        """
        super().__init__()
        self.dropout_t = dropout_dict.get('text_dropout', 0.1)
        self.dropout_g = dropout_dict.get('graph_dropout', 0.1)
        self.dropout_v = dropout_dict.get('vision_dropout', 0.1)
        self.dropout_a = dropout_dict.get('audio_dropout', 0.1)
        self.num_classes = num_classes
        self.use_cross_modal = use_cross_modal

        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        self.norm3 = nn.LayerNorm(hidden_channels)

        # Vision LSTM
        self.vision_lstm = AttentionBiLSTM(
            input_dim=vision_dim, 
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            num_layers=2,
            dropout=self.dropout_v
        )

        # Audio LSTM
        self.audio_lstm = AttentionBiLSTM(
            input_dim=audio_dim, 
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            num_layers=2,
            dropout=self.dropout_a
        )

        self.text_proj = nn.Linear(text_dim, hidden_channels)
        self.dropout_text = nn.Dropout(self.dropout_t)

        if self.use_cross_modal:
            self.cross_modal_attn = CrossModalAttention(
                dim=hidden_channels,
                num_heads=4,
                dropout=self.dropout_g
            )

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_g, add_self_loops=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout_g, add_self_loops=True)

        out_dim = 1 if num_classes == 2 else num_classes
        self.lin = nn.Linear(hidden_channels, out_dim)

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
        x_vision = data.x_vision    # (N_vision, seq_len, v_Dim)
        x_audio = data.x_audio      # (N_audio, seq_len, a_Dim)
        node_types = data.node_types
        vision_lengths = data.vision_lengths
        audio_lengths = data.audio_lengths

        device = x.device

        text_features = self.dropout_text(self.text_proj(x)) # (N_total, Hidden)
        final_x = text_features.clone()

        flat_node_types = []
        if isinstance(node_types[0], list):
            for sublist in node_types: flat_node_types.extend(sublist)
        else:
            flat_node_types = node_types

        # Vision
        vision_features = None
        vision_indices = []
        if x_vision.size(0) > 0:
            h_vision, _ = self.vision_lstm(x_vision, vision_lengths) # (N, Hidden)
            vision_indices = [i for i, t in enumerate(flat_node_types) if t == 'vision']
            if len(vision_indices) > 0:
                final_x[vision_indices] = h_vision
                vision_features = h_vision

        # Audio
        audio_features = None
        audio_indices = []
        if x_audio.size(0) > 0:
            h_audio, _ = self.audio_lstm(x_audio, audio_lengths)
            audio_indices = [i for i, t in enumerate(flat_node_types) if t == 'audio']
            
            if len(audio_indices) > 0:
                final_x[audio_indices] = h_audio
                audio_features = h_audio

        if self.use_cross_modal:
            # Text node indices (summary, topic, transcription)
            text_indices = [i for i, t in enumerate(flat_node_types) 
                          if t in ['summary', 'topic', 'transcription']]
            
            modal_features_list = []
            if vision_features is not None:
                modal_features_list.append(vision_features)
            if audio_features is not None:
                modal_features_list.append(audio_features)

            # Cross Attention
            if len(text_indices) > 0 and len(modal_features_list) > 0:
                modal_features = torch.cat(modal_features_list, dim=0)  # (N_modal, Hidden)
                text_feats = final_x[text_indices]  # (N_text, Hidden)

                enhanced_text = self.cross_modal_attn(text_feats, modal_features)
                final_x[text_indices] = enhanced_text
        
        # GAT 레이어 1
        x = F.dropout(final_x, p=self.dropout_g, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)

        # GAT 레이어 2
        x_in = x
        x = F.dropout(x, p=self.dropout_g, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x + x_in)
        x = F.elu(x)
        
        # GAT 레이어 3
        x = F.dropout(x, p=self.dropout_g, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        
        # batch 벡터를 이용해 그래프별 평균 계산
        # graph_embeddings = global_mean_pool(x, batch) # [batch_size, hidden_channels]

        # Summary node로 최종 로짓값 산출
        summary_nodes = x[data.ptr[:-1]]
        
        # Classification
        # x = F.dropout(graph_embeddings, p=self.dropout, training=self.training)
        out = F.dropout(summary_nodes, p=0.5, training=self.training)
        out = self.lin(out)
        if self.num_classes == 2:
            out = out.squeeze(-1)
        return out
    
    