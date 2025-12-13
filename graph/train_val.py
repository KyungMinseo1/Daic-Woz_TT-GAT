import torch, sys
from tqdm import tqdm
from loguru import logger
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler

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

def train_gat(train_loader, model, criterion, optimizer, device, mode='multimodal', num_classes=2):  
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
    
    if mode=='multimodal':
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