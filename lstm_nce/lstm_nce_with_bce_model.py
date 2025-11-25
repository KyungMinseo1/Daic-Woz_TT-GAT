import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class SimpleBiLSTM_NBCE(nn.Module):
  def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3):
    super().__init__()
    self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_dim*2, output_dim)
    self.bfc = nn.Linear(output_dim, 1)
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
    y_pred = self.bfc(logits)
    return logits, y_pred

class NBCEModel(nn.Module):
  def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.3, temperature=0.07):
    super().__init__()
    self.encoder_q = SimpleBiLSTM_NBCE(input_dim, hidden_dim, output_dim, num_layers, dropout)
    self.encoder_k = SimpleBiLSTM_NBCE(input_dim, hidden_dim, output_dim, num_layers, dropout)

    # EMA momentum coefficient
    self.T = temperature

    # initialize key encoder same as query encoder
    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
      param_k.data.copy_(param_q.data)
      param_k.requires_grad = False  # target encoder는 gradient X

  def forward(self, q, q_len, q_label, queue, queue_labels, criterion, lmbda):
    """
    Args:
      q: (B, seq_len, input_dim) - query sequences
      q_len: (B,) - query lengths
      q_label: (B,) - query labels
      queue: (D, K) - queue features
      queue_labels: (K,) - queue labels
      criterion: loss function (BCELoss OR BCEWithLogitsLoss)
      lmbda: weight lambda (high lambda -> focus more on bce_loss)
    """
    batch_size = q.size(0)

    q_logits, q_pred = self.encoder_q(q, q_len)

    q_feat = F.normalize(q_logits, dim=1)  # (B, D)

    if isinstance(criterion, nn.BCELoss):
      q_prob = nn.Sigmoid(q_pred)
      bce_loss = criterion(q_prob, q_label)
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
      bce_loss = criterion(q_pred, q_label)
    else:
      print("No Criterion: Skipping BCE_Loss")

    correct_sum = ((torch.sigmoid(q_pred) > 0.5).int() == q_label).float().sum()

    with torch.no_grad():
      k_logits, _ = self.encoder_k(q, q_len)
      k_feat = F.normalize(k_logits, dim=1)  # (B, D)

    # extract queue features from k_model (queue_size, D)
    queue_features = queue.clone().detach().T  # (K, D)
    all_keys = torch.cat([k_feat, queue_features], dim=0)  # (B+K, D)
    all_labels = torch.cat([q_label, queue_labels], dim=0)  # (B+K,)

    # Query & Queue similarity: (B, Q)
    logits = torch.matmul(q_feat, all_keys.T) / self.T

    # mask[i,j] = 1 if q_label[i] == queue_labels[j]
    q_labels_expanded = q_label.unsqueeze(1)  # (B, 1)
    all_labels_expanded = all_labels.unsqueeze(0)  # (1, B+K)
    mask = (q_labels_expanded == all_labels_expanded).float()  # (B, B+K)

    # mask out self-contrast (diagonal of first B columns)
    self_mask = torch.eye(batch_size, dtype=torch.float, device=q.device)
    self_mask = F.pad(self_mask, (0, all_keys.size(0) - batch_size), value=0)  # (B, B+K)
    mask = mask * (1 - self_mask)

    # for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # calculate log probability
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # mean of positive pairs log probability
    mask_pos_pairs = mask.sum(1)  # 각 query의 positive pair 개수
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    loss = -mean_log_prob_pos.mean()

    return (1-lmbda)*loss+(lmbda)*bce_loss, correct_sum

