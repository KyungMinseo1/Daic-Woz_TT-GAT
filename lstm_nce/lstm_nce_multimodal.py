import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

def initialize_lstm_for_contrastive_learning(module: nn.Module):
  if isinstance(module, nn.LSTM):
    for name, param in module.named_parameters():
      if 'weight' in name:
        nn.init.xavier_uniform_(param.data)
          
      elif 'bias' in name:
        param.data.fill_(0)
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)
              
  elif isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight.data)
    if module.bias is not None:
      module.bias.data.fill_(0)

class SimpleBiLSTM(nn.Module):
  def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3):
    super().__init__()
    self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_dim*2, output_dim)
  def forward(self, x, lengths):
    """
    Args:
      x: (batch, seq_len, input_dim) - already feature vectors
      lengths: (batch,) - sequence lengths
    """
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
  

class NCEMultimodalModel(nn.Module):
  def __init__(self, x_input_dim:int, x2_input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3):
    super().__init__()
    self.encoder_x_q = SimpleBiLSTM(x_input_dim, hidden_dim, output_dim, num_layers, dropout)
    self.encoder_x2_q = SimpleBiLSTM(x2_input_dim, hidden_dim, output_dim, num_layers, dropout)
    self.encoder_x_k = SimpleBiLSTM(x_input_dim, hidden_dim, output_dim, num_layers, dropout)
    self.encoder_x2_k = SimpleBiLSTM(x2_input_dim, hidden_dim, output_dim, num_layers, dropout)

    initialize_lstm_for_contrastive_learning(self.encoder_x_q)
    initialize_lstm_for_contrastive_learning(self.encoder_x2_q)

    # initialize key encoder same as query encoder
    for param_q, param_k in zip(self.encoder_x_q.parameters(), self.encoder_x_k.parameters()):
      param_k.data.copy_(param_q.data)
      param_k.requires_grad = False  # target encoder는 gradient X
    for param_q, param_k in zip(self.encoder_x2_q.parameters(), self.encoder_x2_k.parameters()):
      param_k.data.copy_(param_q.data)
      param_k.requires_grad = False  # target encoder는 gradient X

  def forward(self, x, x_len, x2, x2_len, x_queue, x2_queue):
    """
    Args:
      x: (B, seq_len, input_dim) - x(transcription) sequences
      x_len: (B,) - x(transcription) lengths
      x2: (B, seq_len, input_dim) - x2(vision or audio) sequences
      x2_len: (B,) - x2(vision or audio) lengths
      x_queue: (C, K) - x queue features
      x2_queue: (C, K) - x2 queue features
    """
    B = x.size(0) # Extract batch size

    q = F.normalize(self.encoder_x_q(x, x_len)[0], dim=1)  # (B, C) -> transcription LSTM feat
    q2 = F.normalize(self.encoder_x2_q(x2, x2_len)[0], dim=1)  # (B, C) -> vision or audio LSTM feat

    C = q.size(1)

    with torch.no_grad():
      k = F.normalize(self.encoder_x_k(x, x_len)[0], dim=1)  # (B, C) -> transcription LSTM feat
      k2 = F.normalize(self.encoder_x2_k(x2, x2_len)[0], dim=1)  # (B, C) -> vision or audio LSTM feat

    # q: transcription, k: vision
    # Positive logits
    l1_pos = torch.bmm(q.view(B, 1, C), k2.view(B, C, 1)).squeeze(2) # (B, 1)
    # Negative logits
    l1_neg = torch.mm(q, x2_queue) # (B, K)
    # logits
    logits1 = torch.cat([l1_pos, l1_neg], dim=1) # (B, 1+K)
    # contrastive loss
    labels1 = torch.zeros(B, device=logits1.device, dtype=torch.long)

    # q: vision, k: transcription
    # Positive logits
    l2_pos = torch.bmm(q2.view(B, 1, C), k.view(B, C, 1)).squeeze(2) # (B, 1)
    # Negative logits
    l2_neg = torch.mm(q2, x_queue) # (B, K)
    # logits
    logits2 = torch.cat([l2_pos, l2_neg], dim=1) # (B, 1+K)
    # contrastive loss
    labels2 = torch.zeros(B, device=logits2.device, dtype=torch.long)

    return (logits1, labels1), (logits2, labels2), (k, k2), (q.std(dim=0).mean(), q2.std(dim=0).mean())
  


