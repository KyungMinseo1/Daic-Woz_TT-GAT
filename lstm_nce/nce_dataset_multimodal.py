import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LSTM_NCE_Multimodal_DATASET(Dataset):
  def __init__(self, x, x2):
    """
    x: list of transcription data (id_num*group_num, seq_len, LM_dim)
    x2: list of temporal data (id_num*group_num, seq_len, feature_columns)
    """
    super().__init__()
    self.x = x
    self.x2 = x2

  def __getitem__(self, idx):
    x = self.x[idx]
    x2 = self.x2[idx]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)

  def __len__(self):
    return len(self.x)