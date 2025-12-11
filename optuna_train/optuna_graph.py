import optuna
import sys, os, argparse, yaml, path_config
from collections import Counter
import pandas as pd
from loguru import logger

import torch
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader

from ..graph.multimodal_bilstm.GAT import GATClassifier as BiLSTMGAT
from ..graph.multimodal_proxy.GAT import GATClassifier as ProxyGAT
from ..graph.multimodal_topic_bilstm.GAT import GATClassifier as TopicBiLSTMGAT
from ..graph.multimodal_topic_bilstm_proxy.GAT import GATClassifier as TopicProxyBiLSTMGAT
from ..graph.multimodal_topic_proxy.GAT import GATClassifier as TopicProxyGAT

from ..graph.multimodal_bilstm.dataset import make_graph as BiLSTM_make_graph
from ..graph.multimodal_proxy.dataset import make_graph as Proxy_make_graph
from ..graph.multimodal_topic_bilstm.dataset import make_graph as TopicBiLSTM_make_graph
from ..graph.multimodal_topic_bilstm_proxy.dataset import make_graph as TopicProxyBiLSTM_make_graph
from ..graph.multimodal_topic_proxy.dataset import make_graph as TopicProxy_make_graph

from ..graph.train_val import train_gat, validation_gat

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
  'multimodal_topic_proxy':TopicProxyGAT
}

MAKE_GRAPH = {
  'multimodal_bilstm':BiLSTM_make_graph,
  'multimodal_proxy':Proxy_make_graph,
  'multimodal_topic_bilstm':TopicBiLSTM_make_graph,
  'multimodal_topic_bilstm_proxy':TopicProxyBiLSTM_make_graph,
  'multimodal_topic_proxy':TopicProxy_make_graph
}

def bilstm_objective(
    trial, config, mode,
    train_loader, val_loader, criterion,
    text_dim, vision_dim, audio_dim,
    epochs, device, checkpoints_dir, patience
    ): # with Attention
  
  lr_min = min(config['training']['lr_list']); lr_max = max(config['training']['lr_list'])
  wd_min = min(config['training']['weight_decay_list']); wd_max = max(config['training']['weight_decay_list'])
  nl_min = min(config['model']['num_layers_list']); nl_max = max(config['model']['num_layers_list'])
  do_min = min(config['model']['dropout_list']); do_max = max(config['model']['dropout_list'])
  
  lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
  weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
  optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "MomentumSGD"])
  num_layers = trial.suggest_int("num_layers", nl_min, nl_max)
  dropout = trial.suggest_float("dropout", do_min, do_max)
  use_attention = trial.suggest_bool("use_attention")
  use_text_proj = trial.suggest_bool("use_text_proj")

  dropout_dict = {
    'text_dropout':dropout,
    'graph_dropout':dropout,
    'vision_dropout':dropout,
    'audio_dropout':dropout
  }

  model = MODEL[mode](
    text_dim=text_dim,
    vision_dim=vision_dim,
    audio_dim=audio_dim,
    hidden_channels=256,
    num_layers=num_layers,
    num_classes=2,
    dropout_dict=dropout_dict,
    heads=8,
    use_attention=use_attention,
    use_summary_node=True,
    use_text_proj=use_text_proj
  ).to(device)

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
      logger.info(f"New best model found! (Trial {trial.number})")
      torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"best_model.pth"))
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
    
  return float(val_f1)

def objective(
    trial, config, mode,
    train_loader, val_loader, criterion,
    text_dim, vision_dim, audio_dim,
    epochs, device, checkpoints_dir, patience
    ): # with Attention
  
  lr_min = min(config['training']['lr_list']); lr_max = max(config['training']['lr_list'])
  wd_min = min(config['training']['weight_decay_list']); wd_max = max(config['training']['weight_decay_list'])
  nl_min = min(config['model']['num_layers_list']); nl_max = max(config['model']['num_layers_list'])
  do_min = min(config['model']['dropout_list']); do_max = max(config['model']['dropout_list'])
  
  lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
  weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
  optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "MomentumSGD"])
  num_layers = trial.suggest_int("num_layers", nl_min, nl_max)
  dropout = trial.suggest_float("dropout", do_min, do_max)
  use_text_proj = trial.suggest_bool("use_text_proj")

  dropout_dict = {
    'text_dropout':dropout,
    'graph_dropout':dropout,
    'vision_dropout':dropout,
    'audio_dropout':dropout
  }

  model = MODEL[mode](
    text_dim=text_dim,
    vision_dim=vision_dim,
    audio_dim=audio_dim,
    hidden_channels=256,
    num_layers=num_layers,
    num_classes=2,
    dropout_dict=dropout_dict,
    heads=8,
    use_summary_node=True,
    use_text_proj=use_text_proj
  ).to(device)

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
      logger.info(f"New best model found! (Trial {trial.number})")
      torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"best_model.pth"))
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
    best_value = current_study.best_value

    if val_f1 > best_value:
      logger.info(f"New best model found! (Trial {trial.number})")
      torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"best_model.pth"))
  except ValueError: 
    # first_trial
    logger.info(f"First trial saved. (Trial {trial.number})")
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"best_model_trial_{trial.number}.pth"))
    
  return float(val_f1)


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
                      help="Model for optuna ['multimodal_bilstm', 'multimodal_proxy', 'multimodal_topic_bilstm', 'multimodal_topic_bilstm_proxy', 'multimodal_topic_proxy']")
  
  opt = parser.parse_args()
  logger.info(opt)

  assert opt.mode is not None, logger.error("Need to clarify your model: 'multimodal_bilstm' | 'multimodal_proxy' | 'multimodal_topic_bilstm' | 'multimodal_topic_bilstm_proxy' | 'multimodal_topic_proxy'")

  with open(os.path.join(path_config.ROOT_DIR, opt.optuna_config), 'r', encoding="utf-8") as ymlfile:
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

  # test_id = test_df.Participant_ID.tolist()
  # test_label = test_df.PHQ_Binary.tolist()

  logger.info(f"Processing Train Data (Mode: {opt.mode})")

  train_graphs, t_dim, v_dim, a_dim = MAKE_GRAPH[opt.mode](
    ids = train_id,
    labels = train_label,
    model_name = config['training']['embed_model'],
    colab_path = opt.colab_path,
    use_summary_node = True,
    v_a_connect=False
  )

  logger.info("Processing Validation Data")
  val_graphs, _, _ = MAKE_GRAPH[opt.mode](
    ids = val_id,
    labels = val_label,
    model_name = config['training']['embed_model'],
    colab_path = opt.colab_path,
  	use_summary_node = True,
    v_a_connect=False
  )

  logger.info("__TRAINING_STATS__")
  train_counters = Counter(label.y.item() for label in train_graphs)
  logger.info(train_counters)

  class_weights = train_counters[0] / train_counters[1]
  logger.info(f"Weights: {class_weights}")

  logger.info("__VALIDATION_STATS__")
  val_counters = Counter(label.y.item() for label in val_graphs)
  logger.info(val_counters)

  logger.info("Setting Training Environment")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(device))
  train_loader = DataLoader(train_graphs, batch_size=config['training']['bs'], shuffle=True)
  val_loader = DataLoader(val_graphs, batch_size=config['training']['bs'], shuffle=False)
  logger.info("Environment Ready")

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
            train_loader=train_loader, val_loader=val_loader, criterion=criterion,
            text_dim=t_dim, vision_dim=v_dim, audio_dim=a_dim,
            epochs=opt.num_epochs, device=device, checkpoints_dir=CHECKPOINTS_DIR, patience=opt.patience
          ),
        n_trials=50
      )
    else:
      study.optimize(
          lambda trial: objective(
            trial=trial, config=config, mode=opt.mode,
            train_loader=train_loader, val_loader=val_loader, criterion=criterion,
            text_dim=t_dim, vision_dim=v_dim, audio_dim=a_dim,
            epochs=opt.num_epochs, device=device, checkpoints_dir=CHECKPOINTS_DIR, patience=opt.patience
          ),
        n_trials=50
      )
  except Exception as e:
    logger.error(f"Error in processing optuna: {e}")

  logger.info("Best params:", study.best_params)
  logger.info("Best accuracy:", study.best_value)

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

# ex) python optuna_train/optuna_graph.py --num_epochs 300 --save_dir checkpoints_optuna --save_dir_ multimodal_proxy --patience 50 --mode multimodal_bilstm