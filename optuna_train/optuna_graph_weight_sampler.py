import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import optuna
import os, argparse, yaml, path_config, shutil
from collections import Counter
import pandas as pd
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

from graph.multimodal_bilstm.GAT import GATClassifier as BiLSTMGAT
from graph.multimodal_proxy.GAT import GATClassifier as ProxyGAT
from graph.multimodal_topic_bilstm.GAT import GATClassifier as TopicBiLSTMGAT
from graph.multimodal_topic_bilstm_proxy.GAT import GATClassifier as TopicProxyBiLSTMGAT
from graph.multimodal_topic_proxy.GAT import GATClassifier as TopicProxyGAT
from graph.unimodal_topic.GAT import GATClassifier as UniTopicGAT
from graph.unimodal_topic_v2.GAT import GATClassifier as UniTopicV2GAT

from graph.multimodal_bilstm.dataset import make_graph as BiLSTM_make_graph
from graph.multimodal_proxy.dataset import make_graph as Proxy_make_graph
from graph.multimodal_topic_bilstm.dataset import make_graph as TopicBiLSTM_make_graph
from graph.multimodal_topic_bilstm_proxy.dataset import make_graph as TopicProxyBiLSTM_make_graph
from graph.multimodal_topic_proxy.dataset import make_graph as TopicProxy_make_graph
from graph.unimodal_topic.dataset import make_graph as UniTopic_make_graph
from graph.unimodal_topic_v2.dataset import make_graph as UniTopicV2_make_graph

from graph.train_val import train_gat, validation_gat

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

MODEL = {
  'multimodal_bilstm':BiLSTMGAT,
  'multimodal_proxy':ProxyGAT,
  'multimodal_topic_bilstm':TopicBiLSTMGAT,
  'multimodal_topic_bilstm_proxy':TopicProxyBiLSTMGAT,
  'multimodal_topic_proxy':TopicProxyGAT,
  'unimodal_topic':UniTopicGAT,
  'unimodal_topic_v2':UniTopicV2GAT
}

MAKE_GRAPH = {
  'multimodal_bilstm':BiLSTM_make_graph,
  'multimodal_proxy':Proxy_make_graph,
  'multimodal_topic_bilstm':TopicBiLSTM_make_graph,
  'multimodal_topic_bilstm_proxy':TopicProxyBiLSTM_make_graph,
  'multimodal_topic_proxy':TopicProxy_make_graph,
  'unimodal_topic':UniTopic_make_graph,
  'unimodal_topic_v2':UniTopicV2_make_graph
}

class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, inputs, targets):
    # inputs: Logits (Sigmoid 통과 전)
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss) # pt: 모델이 해당 클래스일 확률
    
    # Focal Term: (1 - pt)^gamma
    F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

    if self.reduction == 'mean':
      return torch.mean(F_loss)
    elif self.reduction == 'sum':
      return torch.sum(F_loss)
    else:
      return F_loss

def bilstm_objective(
    trial, config, mode,
    train_graphs, val_graphs, sampler,
    text_dim, vision_dim, audio_dim,
    epochs, device, checkpoints_dir, patience
    ): # with Attention
  
  lr_list = [float(i) for i in config['training']['lr_list']]
  bs_list = [int(i) for i in config['training']['bs_list']]
  wd_list = [float(i) for i in config['training']['weight_decay_list']]
  gm_list = [float(i) for i in config['training']['focal_gamma_list']]
  nl_list = [int(i) for i in config['model']['num_layers_list']]
  t_do_list = [float(i) for i in config['model']['text_dropout_list']]
  g_do_list = [float(i) for i in config['model']['graph_dropout_list']]
  v_do_list = [float(i) for i in config['model']['vision_dropout_list']]
  a_do_list = [float(i) for i in config['model']['audio_dropout_list']]

  lr_min = min(lr_list); lr_max = max(lr_list)
  wd_min = min(wd_list); wd_max = max(wd_list)
  gm_min = min(gm_list); gm_max = max(gm_list)
  nl_min = min(nl_list); nl_max = max(nl_list)
  t_do_min = min(t_do_list); t_do_max = max(t_do_list)
  g_do_min = min(g_do_list); g_do_max = max(g_do_list)
  v_do_min = min(v_do_list); v_do_max = max(v_do_list)
  a_do_min = min(a_do_list); a_do_max = max(a_do_list)

  lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
  bs = trial.suggest_categorical("batch_size", bs_list)
  weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
  focal_gamma = trial.suggest_float("focal_gamma", gm_min, gm_max)
  optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "MomentumSGD"])
  num_layers = trial.suggest_int("num_layers", nl_min, nl_max)
  t_dropout = trial.suggest_float("t_dropout", t_do_min, t_do_max)
  g_dropout = trial.suggest_float("g_dropout", g_do_min, g_do_max)
  v_dropout = trial.suggest_float("v_dropout", v_do_min, v_do_max)
  a_dropout = trial.suggest_float("a_dropout", a_do_min, a_do_max)
  use_attention = trial.suggest_categorical("use_attention", [True, False])
  use_text_proj = trial.suggest_categorical("use_text_proj", [True, False])

  dropout_dict = {
    'text_dropout':t_dropout,
    'graph_dropout':g_dropout,
    'vision_dropout':v_dropout,
    'audio_dropout':a_dropout
  }

  train_loader = DataLoader(train_graphs, batch_size=bs, sampler=sampler, shuffle=False) # Using Sampler -> shuffle False
  val_loader = DataLoader(val_graphs, batch_size=bs, shuffle=False)

  model = MODEL[mode](
    text_dim=text_dim,
    vision_dim=vision_dim,
    audio_dim=audio_dim,
    hidden_channels=256 if use_text_proj else text_dim,
    num_layers=num_layers,
    num_classes=2,
    dropout_dict=dropout_dict,
    heads=8,
    use_attention=use_attention,
    use_summary_node=True,
    use_text_proj=use_text_proj
  ).to(device)

  criterion = FocalLoss(alpha=focal_gamma, gamma=2.0).to(device)

  if optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "MomentumSGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)

  if config['training']['scheduler'] == 'steplr':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
  elif config['training']['scheduler'] == 'cosinelr':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  elif config['training']['scheduler'] == 'reducelr':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

  logger.info(f"Model initialized with:")
  logger.info(f"  - Text dim: {text_dim}")
  logger.info(f"  - Vision dim: {vision_dim}")
  logger.info(f"  - Audio dim: {audio_dim}")

  logger.info("Training Start")

  best_val_f1 = -1.0
  patience_counter = 0

  temp_path = os.path.join(checkpoints_dir, f"temp_best_trial_{trial.number}.pth")

  try:
    for epoch in range(epochs):
      train_loss, train_acc, train_f1 = train_gat(
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=2
      )

      torch.cuda.empty_cache()

      val_acc, val_f1 = validation_gat(
        val_loader=val_loader,
        model=model,
        device=device,
        num_classes=2
      )

      trial.report(val_f1, epoch)
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
      
      if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        logger.info(f"New best model found! (Trial {trial.number}: F1 {best_val_f1})")
        torch.save(model.state_dict(), temp_path)
        patience_counter = 0
      else:
        patience_counter += 1

      if patience_counter >= patience:
        logger.info(f"Early stopping at epoch {epoch} (Best F1: {best_val_f1})")
        break
      
      if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step(float(val_f1))
      else:
        scheduler.step()

    current_study = trial.study
    try:
      global_best_value = current_study.best_value # 이전 Trial들까지의 최고 기록
    except ValueError:
      global_best_value = -1.0

    if best_val_f1 > global_best_value:
      logger.info(f"Global Best Model Updated! F1: {best_val_f1} (Trial {trial.number})")
      best_path = os.path.join(checkpoints_dir, f"best_model.pth")

      if os.path.exists(temp_path):
        shutil.copy(temp_path, best_path)

  except optuna.exceptions.TrialPruned:
    raise

  finally:
    if os.path.exists(temp_path):
      try:
        os.remove(temp_path)
      except PermissionError:
        pass

  return float(best_val_f1)

def objective(
    trial, config, mode,
    train_graphs, val_graphs, sampler,
    text_dim, vision_dim, audio_dim,
    epochs, device, checkpoints_dir, patience
    ): # with Attention
  
  lr_list = [float(i) for i in config['training']['lr_list']]
  bs_list = [int(i) for i in config['training']['bs_list']]
  wd_list = [float(i) for i in config['training']['weight_decay_list']]
  gm_list = [float(i) for i in config['training']['focal_gamma_list']]
  nl_list = [int(i) for i in config['model']['num_layers_list']]
  t_do_list = [float(i) for i in config['model']['text_dropout_list']]
  g_do_list = [float(i) for i in config['model']['graph_dropout_list']]
  v_do_list = [float(i) for i in config['model']['vision_dropout_list']]
  a_do_list = [float(i) for i in config['model']['audio_dropout_list']]

  lr_min = min(lr_list); lr_max = max(lr_list)
  wd_min = min(wd_list); wd_max = max(wd_list)
  gm_min = min(gm_list); gm_max = max(gm_list)
  nl_min = min(nl_list); nl_max = max(nl_list)
  t_do_min = min(t_do_list); t_do_max = max(t_do_list)
  g_do_min = min(g_do_list); g_do_max = max(g_do_list)
  v_do_min = min(v_do_list); v_do_max = max(v_do_list)
  a_do_min = min(a_do_list); a_do_max = max(a_do_list)
  
  lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
  bs = trial.suggest_categorical("batch_size", bs_list)
  weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
  focal_gamma = trial.suggest_float("focal_gamma", gm_min, gm_max)
  optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "MomentumSGD"])
  num_layers = trial.suggest_int("num_layers", nl_min, nl_max)
  t_dropout = trial.suggest_float("t_dropout", t_do_min, t_do_max)
  g_dropout = trial.suggest_float("g_dropout", g_do_min, g_do_max)
  v_dropout = trial.suggest_float("v_dropout", v_do_min, v_do_max)
  a_dropout = trial.suggest_float("a_dropout", a_do_min, a_do_max)
  use_text_proj = trial.suggest_categorical("use_text_proj", [True, False])

  dropout_dict = {
    'text_dropout':t_dropout,
    'graph_dropout':g_dropout,
    'vision_dropout':v_dropout,
    'audio_dropout':a_dropout
  }

  train_loader = DataLoader(train_graphs, batch_size=bs, sampler=sampler, shuffle=False) # Using Sampler -> shuffle False
  val_loader = DataLoader(val_graphs, batch_size=bs, shuffle=False)

  if 'unimodal' in mode:
    model = MODEL[mode](
      text_dim=text_dim,
      hidden_channels=256 if use_text_proj else text_dim,
      num_layers=num_layers,
      num_classes=2,
      dropout_dict=dropout_dict,
      heads=8,
      use_summary_node=True,
      use_text_proj=use_text_proj
    ).to(device)
  else:
    model = MODEL[mode](
      text_dim=text_dim,
      vision_dim=vision_dim,
      audio_dim=audio_dim,
      hidden_channels=256 if use_text_proj else text_dim,
      num_layers=num_layers,
      num_classes=2,
      dropout_dict=dropout_dict,
      heads=8,
      use_summary_node=True,
      use_text_proj=use_text_proj
    ).to(device)

  criterion = FocalLoss(alpha=focal_gamma, gamma=2.0).to(device)

  if optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "MomentumSGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)

  if config['training']['scheduler'] == 'steplr':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
  elif config['training']['scheduler'] == 'cosinelr':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  elif config['training']['scheduler'] == 'reducelr':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

  logger.info(f"Model initialized with:")
  logger.info(f"  - Text dim: {text_dim}")
  logger.info(f"  - Vision dim: {vision_dim}")
  logger.info(f"  - Audio dim: {audio_dim}")

  logger.info("Training Start")

  best_val_f1 = -1
  patience_counter = 0
  
  temp_path = os.path.join(checkpoints_dir, f"temp_best_trial_{trial.number}.pth")

  try:
    for epoch in range(epochs):
      if 'unimodal' in mode:
        train_loss, train_acc, train_f1 = train_gat(
          train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          device=device,
          mode='unimodal',
          num_classes=2
        )
      else:
        train_loss, train_acc, train_f1 = train_gat(
          train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          device=device,
          num_classes=2
        )

      torch.cuda.empty_cache()

      val_acc, val_f1 = validation_gat(
        val_loader=val_loader,
        model=model,
        device=device,
        num_classes=2
      )

      trial.report(val_f1, epoch)
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
      
      if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        logger.info(f"New best model found! (Trial {trial.number}: F1 {best_val_f1})")
        torch.save(model.state_dict(), temp_path)
        patience_counter = 0
      else:
        patience_counter += 1

      if patience_counter >= patience:
        logger.info(f"Early stopping at epoch {epoch} (Best F1: {best_val_f1})")
        break
      
      if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step(float(val_f1))
      else:
        scheduler.step()

    current_study = trial.study
    try:
      global_best_value = current_study.best_value # 이전 Trial들까지의 최고 기록
    except ValueError:
      global_best_value = -1.0

    if best_val_f1 > global_best_value:
      logger.info(f"Global Best Model Updated! F1: {best_val_f1} (Trial {trial.number})")
      best_path = os.path.join(checkpoints_dir, f"best_model.pth")

      if os.path.exists(temp_path):
        shutil.copy(temp_path, best_path)

  except optuna.exceptions.TrialPruned:
    raise

  finally:
    if os.path.exists(temp_path):
      try:
        os.remove(temp_path)
      except PermissionError:
        pass

  return float(best_val_f1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', default=100, type=int,
                      help='Number of training epochs.')
  parser.add_argument('--optuna_config', type=str, default='optuna_search_grid.yaml',
                      help="Which optuna configuration to use. See into 'config' folder.")
  # If you want to resume, just put in the original directory path with study SQlite DB.
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
  parser.add_argument('--mode', type=str, default=None,
                      help=f"Model for optuna {list(MODEL.keys())}")
  parser.add_argument('--tt_connect', type=bool, default=False,
                      help="Text to text connection.")
  
  opt = parser.parse_args()
  logger.info(opt)

  assert opt.mode is not None, logger.error(f"Need to clarify your model. Should be in {list(MODEL.keys())}")

  with open(os.path.join(path_config.BASE_DIR, opt.optuna_config), 'r', encoding="utf-8") as ymlfile:
    config = yaml.safe_load(ymlfile)

  os.makedirs(opt.save_dir, exist_ok=True)
  CHECKPOINTS_DIR = os.path.join(opt.save_dir, opt.save_dir_)
  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
  LOGS_DIR = os.path.join(opt.save_dir, opt.save_dir_, 'logs')
  os.makedirs(LOGS_DIR, exist_ok=True)

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

  logger.info(f"Processing Train Data (Mode: {opt.mode})")

  train_graphs, dim_list = MAKE_GRAPH[opt.mode](
    ids = train_id+val_id,
    labels = train_label+val_label,
    model_name = config['training']['embed_model'],
    colab_path = opt.colab_path,
    use_summary_node = True,
    t_t_connect=opt.tt_connect,
    v_a_connect=False
  )

  logger.info("Processing Validation Data")
  val_graphs, _ = MAKE_GRAPH[opt.mode](
    ids = test_id,
    labels = test_label,
    model_name = config['training']['embed_model'],
    colab_path = opt.colab_path,
    use_summary_node = True,
    t_t_connect=opt.tt_connect,
    v_a_connect=False
  )

  if 'unimodal' in opt.mode:
    # temporary value
    t_dim = dim_list
    v_dim = None
    a_dim = None
  else:
    t_dim = dim_list[0]
    v_dim = dim_list[1]
    a_dim = dim_list[2]

  logger.info("__TRAINING_STATS__")
  train_targets = [label.y.item() for label in train_graphs]
  train_counters = Counter(train_targets)
  logger.info(train_counters)

  weight_for_0 = 1.0 / train_counters[0]
  weight_for_1 = 1.0 / train_counters[1]

  logger.info(f"Weights for 0: {weight_for_0} | Weights for 1: {weight_for_1}")

  val_counters = Counter(label.y.item() for label in val_graphs)
  logger.info(val_counters)

  samples_weights = [weight_for_1 if t == 1 else weight_for_0 for t in train_targets]

  sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

  logger.info("Using WeightedRandomSampler for Class Balancing")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(device))

  db_path = os.path.join(LOGS_DIR, "optuna_study.db")
  storage_name = f"sqlite:///{db_path}"
  study_name = opt.mode
  study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    storage=storage_name,
    study_name=study_name,
    load_if_exists=True
  )

  try:
    if 'bilstm' in opt.mode:
      study.optimize(
          lambda trial: bilstm_objective(
            trial=trial, config=config, mode=opt.mode,
            train_graphs=train_graphs, val_graphs=val_graphs, sampler=sampler,
            text_dim=t_dim, vision_dim=v_dim, audio_dim=a_dim,
            epochs=opt.num_epochs, device=device, checkpoints_dir=CHECKPOINTS_DIR, patience=opt.patience
          ),
        n_trials=50
      )
    else:
      study.optimize(
          lambda trial: objective(
            trial=trial, config=config, mode=opt.mode,
            train_graphs=train_graphs, val_graphs=val_graphs, sampler=sampler,
            text_dim=t_dim, vision_dim=v_dim, audio_dim=a_dim,
            epochs=opt.num_epochs, device=device, checkpoints_dir=CHECKPOINTS_DIR, patience=opt.patience
          ),
        n_trials=50
      )
  except Exception as e:
    logger.error(f"Error in processing optuna: {e}")

  logger.info(f"Best Params: {study.best_params}")
  logger.info(f"Best Validation F1: {study.best_value}")

  importance_visual_path = os.path.join(LOGS_DIR, "importance.html")
  history_visual_path = os.path.join(LOGS_DIR, "history.html")

  try:
    # save in HTML format
    fig_imp = optuna.visualization.plot_param_importances(study)
    fig_imp.write_html(importance_visual_path)
    
    fig_hist = optuna.visualization.plot_optimization_history(study)
    fig_hist.write_html(history_visual_path)
    
    logger.info(f"Visualizations saved to {opt.save_dir}")
  except Exception as e:
    logger.error(f"Failed to save visualization: {e}")

if __name__=="__main__":
  main()

# Example for multimodal_proxy
#   -> python optuna_train/optuna_graph_weight_sampler.py --mode multimodal_proxy --num_epochs 300 --patience 30 --save_dir checkpoints_optuna --save_dir_ multimodal_proxy