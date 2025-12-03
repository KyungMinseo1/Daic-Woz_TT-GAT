import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from multiprocessing import Manager
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
from lstm_nce_with_bce_model import NBCEModel
from prepare_dataset_vision_nce import read_v
import pandas as pd
import os, config_path, json, collections

# OpenMP / MKL 충돌 방지용 설정
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from nce_dataset import LSTM_NCE_DATASET
import matplotlib.pyplot as plt
import yaml
import argparse

def collate_batch(batch):
  # batch: list of (tensor(tokens), tensor(label))
  x = [item[0] for item in batch]
  labels = torch.stack([item[1] for item in batch]).long()
  lengths = torch.tensor([t.size(0) for t in x], dtype=torch.long)
  x_padded = pad_sequence(x, batch_first=True, padding_value=0) # pad id = 0
  return x_padded, lengths, labels

@torch.no_grad()
def update_momentum_encoder(encoder_k, encoder_q, m=0.99):
  for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
    param_k.data = param_k.data * m + param_q.data * (1. - m)

class Queue(nn.Module):
  def __init__(self, output_dim:int, queue_size=4096):
    super().__init__()
    self.register_buffer("queue", torch.randn(output_dim, queue_size))
    self.queue = F.normalize(self.queue, dim=0)
    self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))
    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    self.register_buffer("is_initialized", torch.tensor(False))

  @torch.no_grad()
  def initialize_queue(self, keys, labels):
    if not self.is_initialized: # type: ignore
      batch_size = keys.shape[0]
      # repeat with first batch
      for i in range(0, self.queue.shape[1], batch_size):
        end = min(i + batch_size, self.queue.shape[1])
        self.queue[:, i:end] = keys[:end-i].T
        self.queue_labels[i:end] = labels[:end-i] # type: ignore
      self.is_initialized.fill_(True) # type: ignore
  
  @torch.no_grad()
  def dequeue_and_enqueue(self, keys, labels): # labels 인자 추가
    keys = keys.detach()
    batch_size = keys.shape[0]
    ptr = int(self.queue_ptr) # type: ignore
    queue_size = self.queue.shape[1]

    end = (ptr + batch_size) % queue_size
    
    if end > ptr:
      self.queue[:, ptr:end] = keys[:end - ptr].T
      self.queue_labels[ptr:end] = labels[:end - ptr] # type: ignore
    else:
      remaining = queue_size - ptr
      self.queue[:, ptr:] = keys[:remaining].T
      self.queue_labels[ptr:] = labels[:remaining] # type: ignore
      self.queue[:, :end] = keys[remaining:].T
      self.queue_labels[:end] = labels[remaining:] # type: ignore

    self.queue_ptr[0] = end # type: ignore

def train(model, dataloader, queue, optimizer, device, config, criterion, lmbda, warmup=False):
  model.encoder_q.train()
  model.encoder_k.eval()

  total_loss = 0.0
  num_batches = len(dataloader)
  pbar = tqdm(dataloader, desc='Training', ncols=120)
  total_correct = 0
  total_count = 0

  for batch_idx, (x, x_len, y) in enumerate(pbar):
    x, x_len, y = x.to(device), x_len.to(device), y.to(device)

    if warmup:
      with torch.no_grad():
        k_feat, _ = model.encoder_k(x, x_len)
        new_k = F.normalize(k_feat, dim=1)
        queue.dequeue_and_enqueue(new_k, y)
      continue

    loss, correct_sum, q_std = model(x, x_len, y, queue.queue, queue.queue_labels, criterion, lmbda)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute key features (momentum encoder)
    with torch.no_grad():
      update_momentum_encoder(model.encoder_k, model.encoder_q, config['training']['M'])
      k_feat, _ = model.encoder_k(x, x_len)
      new_k = F.normalize(k_feat, dim=1)

    with torch.no_grad():
      queue.dequeue_and_enqueue(new_k, y)

    total_loss += loss.item()
    avg_loss = total_loss / (batch_idx + 1)
    total_correct += correct_sum
    total_count += x.size(0)

    current_lr = optimizer.param_groups[0]['lr']

    total_norm = 0
    for p in model.encoder_q.parameters():
      if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2

    # Update progress bar with detailed info
    pbar.set_postfix({
      'loss': f'{loss.item():.4f}',
      'avg': f'{avg_loss:.4f}',
      'lr': f'{current_lr:.2e}',
      'grad_norm': f'{total_norm**0.5}',
      'feat_std': f'{q_std}',
      'correct_rate':f'{total_correct/total_count:.4f}',
      'batch': f'{batch_idx+1}/{num_batches}'
    })

  if warmup:
    return avg_loss if not warmup else 0.0, _
  else:
    total_correct_rate = total_correct/total_count
    return avg_loss if not warmup else 0.0, total_correct_rate

def validation_knn(model, train_loader, val_loader, device, criterion, lmbda, k=5, batch_size=256):
  model.encoder_k.eval()
  
  print("Building feature bank from training set...")
  train_features = []
  train_labels = []
  with torch.no_grad():
    for x, x_len, y in tqdm(train_loader, desc='Feature Bank', ncols=100):
      x, x_len, y = x.to(device), x_len.to(device), y.to(device)
      k_feat, _ = model.encoder_k(x, x_len)
      features = F.normalize(k_feat, dim=1)
      train_features.append(features.cpu())  # CPU로 이동하여 메모리 절약
      train_labels.append(y.cpu())
  
  train_features = torch.cat(train_features, dim=0)
  train_labels = torch.cat(train_labels, dim=0)

  print(f"Running KNN validation (k={k})...")
  correct = 0
  total = 0
  num_batches = len(val_loader)

  with torch.no_grad():
    pbar = tqdm(val_loader, desc='KNN Validation', ncols=120)
    for batch_idx, (x, x_len, y) in enumerate(pbar):
      x, x_len, y = x.to(device), x_len.to(device), y.to(device)
      
      k_feat, _ = model.encoder_k(x, x_len)
      features = F.normalize(k_feat, dim=1)
      
      batch_predictions = []
      for i in range(0, train_features.size(0), batch_size):
        batch_train_features = train_features[i:i+batch_size].to(device)
        batch_sim = torch.matmul(features, batch_train_features.T)
        batch_predictions.append(batch_sim)
      
      similarities = torch.cat(batch_predictions, dim=1)
      
      # Top-k
      _, topk_indices = similarities.topk(k, dim=1)
      topk_labels = train_labels[topk_indices.cpu()]
      predictions = torch.mode(topk_labels, dim=1)[0]
      
      correct += (predictions.to(device) == y).sum().item()
      total += y.size(0)
      
      pbar.set_postfix({
          'acc': f'{correct/total:.4f}',
          'correct_knn': f'{correct}/{total}',
          'batch': f'{batch_idx+1}/{num_batches}'
      })
  accuracy = correct / total
  print(f"\nKNN Validation Accuracy (k={k}): {accuracy:.4f}")
  return accuracy
  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', default=100, type=int,
                      help='Number of training epochs.')
  parser.add_argument('--workers', default=2, type=int,
                      help='Number of data loader workers.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='Path to latest checkpoint (default: none).')
  parser.add_argument('--config', type=str, default='lstm_nce/configs/architecture.yaml',
                      help="Which configuration to use. See into 'config' folder.")
  parser.add_argument('--save_dir', type=str, default='checkpoints', metavar='PATH',
                      help="Directory path to save model")
  parser.add_argument('--save_dir_', type=str, default='checkpoints1', metavar='PATH',
                      help="Exact directory path to save model")
  parser.add_argument('--patience', type=int, default=10, 
                      help="How many epochs wait before stopping for validation loss not improving.")
  parser.add_argument('--warmup_epochs', type=int, default=3,
                      help='Number of warm-up epochs to fill the queue before full training.')
  
  opt = parser.parse_args()
  print(opt)

  with open(opt.config, 'r', encoding="utf-8") as ymlfile:
    config = yaml.safe_load(ymlfile)

  os.makedirs(opt.save_dir, exist_ok=True)
  CHECKPOINTS_DIR = os.path.join(opt.save_dir, opt.save_dir_)
  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

  history = {
    'epoch':[],
    'train_loss':[],
    'train_correct':[],
    'val_accuracy':[]
  }

  mgr = Manager()
  train_dataset = mgr.list()
  train_label_dataset = mgr.list()
  val_dataset = mgr.list()
  val_label_dataset = mgr.list()
  
  train_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
  val_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'dev_split_Depression_AVEC2017.csv'))
  
  train_id = train_df.Participant_ID.tolist()
  val_id = val_df.Participant_ID.tolist()
  train_label = train_df.PHQ8_Binary.tolist()
  val_label = val_df.PHQ8_Binary.tolist()
  train_zip = zip(train_id, train_label)
  val_zip = zip(val_id, val_label)

  with Pool(processes=10) as p:
    with tqdm(total=len(train_id)) as pbar:
      for v in p.imap_unordered(partial(read_v, dataset=train_dataset, label_dataset=train_label_dataset), train_zip):
        pbar.update()

    with tqdm(total=len(val_id)) as pbar:
      for v in p.imap_unordered(partial(read_v, dataset=val_dataset, label_dataset=val_label_dataset),val_zip):
        pbar.update()

  # train_dataset: (id*B, seq_len, v_columns)
  # train_label_dataset: (id*B,)
  input_dim = len(train_dataset[0][0])
  print("Current Feature Dimenseion:", input_dim)

  print("__TRAINING STATS__")
  train_counters = collections.Counter(label for label in train_label_dataset)
  print(train_counters)
  
  class_weights = train_counters[0] / train_counters[1]
  print("Weights", class_weights)

  print("__VALIDATION STATS__")
  val_counters = collections.Counter(label for label in val_label_dataset)
  print(val_counters)
  print("___________________")


  train_data = LSTM_NCE_DATASET(train_dataset, train_label_dataset)
  del train_dataset
  val_data = LSTM_NCE_DATASET(val_dataset, val_label_dataset)
  del val_dataset

  train_loader = DataLoader(train_data, batch_size=config['training']['bs'], num_workers=config['training']['workers'], shuffle=True, pin_memory=True, collate_fn=collate_batch)
  val_loader = DataLoader(val_data, batch_size=config['training']['bs'], num_workers=config['training']['workers'], shuffle=False, pin_memory=True, collate_fn=collate_batch)
  
  # train setting
  hidden_dim = config['model']['h_dim']
  output_dim = config['model']['o_dim']
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = NBCEModel(input_dim, hidden_dim, output_dim, num_layers=config['model']['depth'], dropout=config['model']['dropout'], temperature=config['training']['T'])
  criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(device))
  queue = Queue(output_dim=output_dim, queue_size=config['training']['q_size'])
  optimizer = torch.optim.SGD(model.encoder_q.parameters(), lr=config['training']['lr'])
  if config['training']['scheduler'] == 'steplr':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
  else:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)

  model.to(device)
  queue.to(device)

  starting_epoch = 0
  if opt.resume and os.path.exists(opt.resume):
    print(f"Loading checkpoint from {opt.resume}")
    checkpoint = torch.load(opt.resume)

    try:
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      queue.load_state_dict(checkpoint['queue_state_dict'])
      starting_epoch = checkpoint['epoch'] + 1
      history = checkpoint['history']
    except Exception as e:
      print("Error Resuming:", e)
      print("Check your configuration files.")
      return
    
    print(f"Resuming from epoch {starting_epoch}")
    print(f"Previous best val_loss: {min(history['val_loss']) if history['val_loss'] else 'N/A'}")
  else:
    print("No checkpoint loaded. Starting from scratch.")

  print("Train Start")

  patience = 0

  for epoch in range(starting_epoch, opt.num_epochs + 1 + opt.warmup_epochs):
    warmup = epoch <= opt.warmup_epochs
    if warmup:
      print(f"\n[Warm-up Epoch {epoch}/{opt.warmup_epochs}] Filling queue only...")
    train_avg_loss, total_correct_rate = train(model=model, dataloader=train_loader, queue=queue, optimizer=optimizer, device=device, config=config, criterion=criterion, lmbda=config['training']['lambda'], warmup=warmup)
    if warmup:
      continue
    val_accuracy = validation_knn(model=model, train_loader=train_loader, val_loader=val_loader, device=device, criterion=criterion, lmbda=config['training']['lambda'])

    history['epoch'].append(epoch)
    history['train_loss'].append(train_avg_loss)
    history['train_correct'].append(total_correct_rate)
    history['val_accuracy'].append(val_accuracy)

    print("#" + str(epoch) + "/" + str(opt.num_epochs) + " loss:" +
                str(train_avg_loss) + " accuracy:" + str(val_accuracy))
    
    checkpoint = {
        'epoch': epoch-opt.warmup_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'queue_state_dict' : queue.state_dict(),
        'config' : config,
        'history': history
    }

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if len(history['val_accuracy']) > 0 and val_accuracy == max(history['val_accuracy']):
      best_path = os.path.join(CHECKPOINTS_DIR, f"best_model.pth")
      torch.save(checkpoint, best_path)
      print(f"Best model saved: {best_path}")
      patience = 0
    else:
      patience += 1
      print(f"Current Patience: {patience}")
    
    if patience >= opt.patience:
      print(f"# {epoch}/{opt.num_epochs} => Early Stopping")
      break

  # 학습 완료 후 그래프 생성
  print("\nGenerating training history plots...")

  # 디렉토리 생성
  plots_dir = os.path.join(CHECKPOINTS_DIR, "plots")
  if not os.path.exists(plots_dir):
      os.makedirs(plots_dir)

  epochs_range = range(starting_epoch, starting_epoch + len(history['train_loss']))

  # Loss 그래프
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.legend()
  plt.grid(True)

  # Accuracy 그래프
  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, history['val_accuracy'], 'r-', label='Val Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Validation Accuracy')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(plots_dir, f'training_history.png'), dpi=300)
  print(f"Plot saved: {os.path.join(plots_dir, f'training_history.png')}")

  # 히스토리를 JSON으로 저장
  history_path = os.path.join(CHECKPOINTS_DIR, f'training_history.json')
  with open(history_path, 'w') as f:
      json.dump(history, f, indent=4)
  print(f"History saved: {history_path}")

  print("\nTraining completed!")
  print(f"Best validation loss: {min(history['val_loss']):.4f}")
  print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")

if __name__=="__main__":
  main()

# first
#   ex) python lstm_nce/train_nce.py
#     -> epochs: 100, workers: 2, config_file: lstm_nce/configs/architecture.yaml, save_path: checkpoints/checkpoints1, patience: 10, warmup_for_queue: 3
# resume
#   ex) python lstm_nce/train_nce.py --resume checkpoints/checkpoints1/best_model.pth