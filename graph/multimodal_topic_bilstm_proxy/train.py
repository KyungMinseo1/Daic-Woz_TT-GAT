import torch, sys, os, argparse, yaml
from .. import path_config
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from collections import Counter
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .GAT import GATClassifier
from .dataset import make_graph


import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

def check_lstm_grad(lstm_module, name="LSTM"):
  total_norm = 0.0
  cnt = 0
  for p_name, p in lstm_module.bilstm.named_parameters():
    if p.grad is not None:
      param_norm = p.grad.data.norm(2)
      total_norm += param_norm.item() ** 2
      cnt += 1
  if cnt > 0:
    total_norm = total_norm ** 0.5
    logger.info(f"{name} Total Grad Norm: {total_norm:.4f}")
  else:
    logger.info(f"{name} Grad: None")

class FocalLoss(torch.nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, inputs, targets):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    
    if self.alpha is not None:
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        f_loss = alpha_t * (1-pt)**self.gamma * bce_loss
    else:
        f_loss = (1-pt)**self.gamma * bce_loss
    
    if self.reduction == 'mean':
      return torch.mean(f_loss)
    return f_loss
  
def train_gat(train_loader, model, criterion, optimizer, device, num_classes=2):  
  model.train()
  total_loss = 0
  all_preds = []
  all_labels = []

  # scaler = GradScaler()

  pbar = tqdm(train_loader, desc='Training', ncols=120)
  
  for idx, batch in enumerate(pbar):
    batch = batch.to(device)

    if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
      logger.error(f"Batch {idx}: Input features (batch.x) contain NaN or Inf! Aborting batch.")
      continue
      
    if torch.isnan(batch.x_vision).any() or torch.isinf(batch.x_vision).any():
      logger.error(f"Batch {idx}: Input features (batch.x_vision) contain NaN or Inf! Aborting batch.")
      continue
      
    if torch.isnan(batch.x_audio).any() or torch.isinf(batch.x_audio).any():
      logger.error(f"Batch {idx}: Input features (batch.x_audio) contain NaN or Inf! Aborting batch.")
      continue

    optimizer.zero_grad()
    
    # with autocast():
    out = model(batch)
    if num_classes == 2:
      # Binary classification
      loss = criterion(out, batch.y.float())
      pred = (torch.sigmoid(out) > 0.5).long()
    else:
      # Multi-class classification
      loss = criterion(out, batch.y)
      pred = out.argmax(dim=1)

    loss.backward()

    # logger.info("--- Gradient Check ---")
    # for name, param in model.named_parameters():
    #   if param.grad is not None:
    #     grad_norm = param.grad.data.norm(2).item()
    #     if grad_norm > 1.0: # 튀는 놈만 출력
    #       logger.info(f"{name}: {grad_norm:.4f}")
    # logger.info("----------------------\n")

    # scaler.scale(loss).backward()
    # scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
    # scaler.step(optimizer)
    # scaler.update()
    optimizer.step()
    
    total_norm = 0.0
    has_nan = False
    zero_grad_cnt = 0
    total_param_cnt = 0

    for name, p in model.named_parameters():
      if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        total_param_cnt += 1
        
        if torch.isnan(param_norm):
          has_nan = True
        if param_norm.item() == 0:
          zero_grad_cnt += 1
  
    total_norm = total_norm ** 0.5 # 전체 모델의 Gradient L2 Norm

    # 만약 Gradient가 아예 끊겼다면(0) 로그 출력
    if total_norm == 0:
      logger.warning(f"Batch {idx}: Total Gradient is ZERO! (Check graph connectivity or detach)")
    
    # 만약 NaN이 떴다면 즉시 중단 권고
    if has_nan:
      logger.error(f"Batch {idx}: Gradient Exploded (NaN detected)!")

    total_loss += loss.item()
    all_preds.extend(pred.cpu().numpy())
    all_labels.extend(batch.y.cpu().numpy())
    
    # 실시간 메트릭 계산 (현재까지 누적)
    current_avg_loss = total_loss / (idx + 1)
    current_accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    
    # Progress bar 업데이트
    pbar.set_postfix({
      'Loss': f'{current_avg_loss:.4f}',
      'Acc': f'{current_accuracy:.4f}',
      'Grad': f'{total_norm:.4f}'
    })
  
  # 최종 메트릭 계산
  avg_loss = total_loss / len(train_loader)
  accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
  
  # F1 score 계산
  if num_classes == 2:
    f1 = f1_score(all_labels, all_preds, average='binary')
  else:
    f1 = f1_score(all_labels, all_preds, average='macro')
  
  return avg_loss, accuracy, f1

def validation_gat(val_loader, model, device, num_classes=2):
  model.eval()
  all_preds = []
  all_labels = []
  
  pbar = tqdm(val_loader, desc='Validation', ncols=120)
  
  with torch.no_grad():
    for batch in pbar:
      batch = batch.to(device)
      out = model(batch)
      
      if num_classes == 2:
        pred = (torch.sigmoid(out) > 0.5).long()
      else:
        pred = out.argmax(dim=1)
      
      all_preds.extend(pred.cpu().numpy())
      all_labels.extend(batch.y.cpu().numpy())
      
      # 실시간 accuracy 업데이트
      current_accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
      pbar.set_postfix({'Acc': f'{current_accuracy:.4f}'})
  
  accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
  
  # F1 score 계산
  if num_classes == 2:
    f1 = f1_score(all_labels, all_preds, average='binary')
  else:
    f1 = f1_score(all_labels, all_preds, average='macro')
  
  return accuracy, f1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', default=100, type=int,
                      help='Number of training epochs.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='Path to latest checkpoint (default: none).')
  parser.add_argument('--config', type=str, default='graph/configs/architecture.yaml',
                      help="Which configuration to use. See into 'config' folder.")
  parser.add_argument('--save_dir', type=str, default='checkpoints', metavar='PATH',
                      help="Directory path to save model")
  parser.add_argument('--save_dir_', type=str, default='checkpoints1', metavar='PATH',
                      help="Exact directory path to save model")
  parser.add_argument('--patience', type=int, default=10, 
                      help="How many epochs wait before stopping for validation loss not improving.")
  parser.add_argument('--sample_rate', type=float, default=1.0, 
                      help="Sampling rate of Data.")
  parser.add_argument('--colab_path', type=str, default=None, 
                      help="Write the path of temporary directory when using colab. (Transcription/Vision/Audio)")
  
  opt = parser.parse_args()
  logger.info(opt)

  with open(os.path.join(path_config.ROOT_DIR, opt.config), 'r', encoding="utf-8") as ymlfile:
    config = yaml.safe_load(ymlfile)

  os.makedirs(opt.save_dir, exist_ok=True)
  CHECKPOINTS_DIR = os.path.join(opt.save_dir, opt.save_dir_)
  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

  history = {
    'epoch':[],
    'train_loss':[],
    'train_acc':[],
    'train_f1':[],
    'val_acc':[],
    'val_f1':[]
  }

  train_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
  val_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'dev_split_Depression_AVEC2017.csv'))
  test_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'full_test_split.csv'))

  train_df = train_df.sample(frac=opt.sample_rate)
  val_df = val_df.sample(frac=opt.sample_rate)
  test_df = test_df.sample(frac=opt.sample_rate)
  
  train_id = train_df.Participant_ID.tolist()
  val_id = val_df.Participant_ID.tolist()
  train_label = train_df.PHQ8_Binary.tolist()
  val_label = val_df.PHQ8_Binary.tolist()

  test_id = test_df.Participant_ID.tolist()
  test_label = test_df.PHQ_Binary.tolist()

  logger.info("Processing Train Data")
  train_graphs, v_dim, a_dim, _ = make_graph(
    ids = train_id+val_id,
    labels = train_label+val_label,
    model_name = config['training']['embed_model'],
    colab_path = opt.colab_path,
    use_summary_node = config['model']['use_summary_node']
  )
  logger.info("Processing Validation Data")
  val_graphs, _, _, _ = make_graph(
    ids = test_id,
    labels = test_label,
    model_name = config['training']['embed_model'],
    colab_path = opt.colab_path,
  	use_summary_node = config['model']['use_summary_node']
  )

  logger.info("__TRAINING_STATS__")
  train_counters = Counter(label.y.item() for label in train_graphs)
  logger.info(train_counters)

  class_weights = train_counters[0] / train_counters[1]
  logger.info(f"Weights: {class_weights}")

  logger.info("__VALIDATION_STATS__")
  val_counters = Counter(label.y.item() for label in val_graphs)
  logger.info(val_counters)

  dropout_dict = {
    'text_dropout':config['model']['t_dropout'],
    'graph_dropout':config['model']['g_dropout'],
    'vision_dropout':config['model']['v_dropout'],
    'audio_dropout':config['model']['a_dropout']
  }

  logger.info("Setting Training Environment")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = GATClassifier(
      text_dim=train_graphs[0].x.shape[1],
      vision_dim=v_dim,
      audio_dim=a_dim,
      hidden_channels=config['model']['h_dim'],
      num_layers=config['model']['num_layers'],
      num_classes=2,
      dropout_dict=dropout_dict,
      heads=config['model']['head'],
      use_attention=config['model']['use_attention'],
      use_summary_node=config['model']['use_summary_node']
  ).to(device)
  logger.info(f"Model initialized with:")
  logger.info(f"  - Text dim: {train_graphs[0].x.shape[1]}")
  logger.info(f"  - Vision dim: {v_dim}")
  logger.info(f"  - Audio dim: {a_dim}")
  logger.info(f"  - Hidden channels: {config['model']['h_dim']}")
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
  if config['training']['scheduler'] == 'steplr':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
  elif config['training']['scheduler'] == 'cosinelr':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
  elif config['training']['scheduler'] == 'reducelr':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(device))
  # criterion = FocalLoss(alpha=0.75, gamma=3.0).to(device)
  logger.info("Environment Ready")
  logger.info("Providing Loader")
  train_loader = DataLoader(train_graphs, batch_size=config['training']['bs'], shuffle=True)
  val_loader = DataLoader(val_graphs, batch_size=config['training']['bs'], shuffle=False)
  logger.info("Loader Ready")

  starting_epoch = 0
  best_val_f1 = -1
  patience_counter = 0
  best_epoch = 0

  if opt.resume and os.path.exists(opt.resume):
    logger.info(f"Loading checkpoint from {opt.resume}")
    checkpoint = torch.load(opt.resume)

    try:
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      starting_epoch = checkpoint['epoch'] + 1
      history = checkpoint['history']
      best_val_f1 = checkpoint['best_val_f1']
    except Exception as e:
      logger.error("Error Resuming:", e)
      logger.error("Check your configuration files.")
      return
    
    logger.info(f"Resuming from epoch {starting_epoch}")
    logger.info(f"Previous best val_f1: {min(history['val_f1']) if history['val_f1'] else 'N/A'}")
  else:
    logger.info("No checkpoint loaded. Starting from scratch.")

  logger.info("Training Start")
  for epoch in range(starting_epoch, starting_epoch + opt.num_epochs + 1):
    train_loss, train_acc, train_f1 = train_gat(
      train_loader=train_loader,
      model=model,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
      num_classes=2)
    
    torch.cuda.empty_cache()
    
    val_acc, val_f1 = validation_gat(
      val_loader=val_loader,
      model=model,
      device=device,
      num_classes=2)
    
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
      scheduler.step(float(val_f1))
    else:
      scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    logger.info(
      f"Epoch {epoch}: "
      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f} | "
      f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f} | "
      f"LR={current_lr:.6f}"
    )

    # if hasattr(model.vision_lstm, 'bilstm') and model.vision_lstm.bilstm.weight_ih_l0.grad is not None:
    #   v_grad_norm = model.vision_lstm.bilstm.weight_ih_l0.grad.norm().item() # type: ignore
    #   logger.info(f"Vision LSTM Grad: {v_grad_norm:.4f}")
    # else:
    #     logger.info("Vision LSTM Grad: None")

    # if hasattr(model.audio_lstm, 'bilstm') and model.audio_lstm.bilstm.weight_ih_l0.grad is not None:
    #   a_grad_norm = model.audio_lstm.bilstm.weight_ih_l0.grad.norm().item() # type: ignore
    #   logger.info(f"Audio LSTM Grad:  {a_grad_norm:.4f}")
    # else:
    #  logger.info("Audio LSTM Grad: None")

    check_lstm_grad(model.vision_lstm, "Vision LSTM")
    check_lstm_grad(model.audio_lstm, "Audio LSTM")

    checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'config' : config,
      'history': history,
      'best_val_f1':best_val_f1
    }

    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      best_epoch = epoch
      patience_counter = 0

      best_path = os.path.join(CHECKPOINTS_DIR, f"best_model.pth")
      torch.save(checkpoint, best_path)
      logger.info(f"Best model saved: {best_path}")
    else:
      patience_counter += 1
      logger.info(f"No improvement for {patience_counter} epoch(s). Best: {best_val_f1:.4f} at epoch {best_epoch}")

    if epoch+1 % 20 == 0:
      checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch{epoch}.pth")
      torch.save(checkpoint, checkpoint_path)
      logger.info(f"Checkpoint saved: {checkpoint_path}")

    if patience_counter >= opt.patience:
      logger.info(f"Early stopping triggered! Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
      break

  logger.info(f"Training finished! Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
  history_df = pd.DataFrame(history)
  history_path = os.path.join(CHECKPOINTS_DIR, 'training_history.csv')
  history_df.to_csv(history_path, index=False)
  logger.info(f"Training history saved to {history_path}")

  # 학습 완료 후 그래프 생성
  logger.info("\nGenerating training history plots...")

  # 디렉토리 생성
  plots_dir = os.path.join(CHECKPOINTS_DIR, "plots")
  if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

  epochs_range = range(starting_epoch, starting_epoch + len(history['train_loss']))

  fig, axes = plt.subplots(2, 2, figsize=(18, 5))

  # 1. Loss 그래프 (첫 번째 칸: axes[0,0])
  axes[0,0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
  axes[0,0].set_xlabel('Epoch')
  axes[0,0].set_ylabel('Loss')
  axes[0,0].set_title('Training Loss')
  axes[0,0].legend()
  axes[0,0].grid(True)

  # 2. Train_F1 그래프 (첫 번째 칸: axes[0,1])
  axes[0,1].plot(epochs_range, history['train_f1'], 'b-', label='Train F1')
  axes[0,1].set_xlabel('Epoch')
  axes[0,1].set_ylabel('F1')
  axes[0,1].set_title('Train F1-score')
  axes[0,1].legend()
  axes[0,1].grid(True)

  # 3. X_std 그래프 (두 번째 칸: axes[1,0])
  axes[1,0].plot(epochs_range, history['val_acc'], 'g-', label='Val Acc') # 색상 구분(green)
  axes[1,0].set_xlabel('Epoch')
  axes[1,0].set_ylabel('Acc')
  axes[1,0].set_title('Validation Accuracy')
  axes[1,0].legend()
  axes[1,0].grid(True)

  # 4. X2_std 그래프 (세 번째 칸: axes[1,1])
  axes[1,1].plot(epochs_range, history['val_f1'], 'r-', label='Val F1') # 색상 구분(red)
  axes[1,1].set_xlabel('Epoch')
  axes[1,1].set_ylabel('F1')
  axes[1,1].set_title('Validation F1-score')
  axes[1,1].legend()
  axes[1,1].grid(True)

  plt.tight_layout()
  
  # 이미지 저장
  plot_path = os.path.join(plots_dir, 'training_history.png')
  plt.savefig(plot_path, dpi=300)

  plt.close(fig) 
  
  logger.info(f"Plot saved: {plot_path}")

if __name__=="__main__":
  main()

# first
#   ex) python graph/train_with_lstm.py --save_dir checkpoints_graph3 --save_dir_ LSTM_graph_11(focal) --num_epochs 150 --patience 30
#     -> epochs: 150, config_file: graph/configs/architecture.yaml, save_path: checkpoints_graph3/LSTM_graph_11(focal), patience: 30
# resume
#   ex) python graph/train.py --resume checkpoints/checkpoints1/best_model.pth

