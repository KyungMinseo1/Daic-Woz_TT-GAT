import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LSTM_NCE_DATASET(Dataset):
  def __init__(self, x, label):
    """
    x: list of temporal data (id_num*group_num, seq_len, feature_columns)
    y: list of labels (id_num*group_num, 1)
    """
    super().__init__()
    self.x = x
    self.y = label

  def __getitem__(self, idx):
    x = self.x[idx]
    y = self.y[idx]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.x)