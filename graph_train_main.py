import optuna
import torch, sys, os, argparse, yaml, path_config
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

from graph.multimodal_bilstm.GAT import GATClassifier as BiLSTMGAT
from graph.multimodal_proxy.GAT import GATClassifier as ProxyGAT
from graph.multimodal_topic_bilstm.GAT import GATClassifier as TopicBiLSTMGAT
from graph.multimodal_topic_bilstm_proxy.GAT import GATClassifier as TopicProxyBiLSTMGAT
from graph.multimodal_topic_proxy.GAT import GATClassifier as TopicProxyGAT

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

def bilstm_objective(
    trial,
    config,
    mode,
    text_dim,
    vision_dim,
    audio_dim
    ): # with Attention
  lr = trial.suggest_float("lr", config['training']['lr_list'][0], config['training']['lr_list'][1])
  weight_decay = trial.suggest_float("weight_decay", config['training']['weight_decay_list'][0], config['training']['weight_decay_list'][1]) # change
  optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "MomentumSGD"])
  num_layers = trial.suggest_int("num_layers", config['model']['num_layers_list'][0], config['model']['num_layers_list'][1])
  dropout = trial.suggest_float("dropout", config['model']['dropout_list'][0], config['model']['dropout_list'][1])
  use_attention = trial.suggest_bool("use_attention")
  use_summary_node = trial.suggest_bool("use_summary_node")

  dropout_dict = {
    'text_dropout':dropout,
    'graph_dropout':dropout,
    'vision_dropout':dropout,
    'audio_dropout':dropout
  }

  if mode == "with_topic":
    logger.info("Starting With Topic BiLSTM")
    model = TopicBiLSTMGAT(
      text_dim=text_dim,
      vision_dim=vision_dim,
      audio_dim=audio_dim,
      hidden_channels=256,
      num_layers=num_layers,
      num_classes=2,
      dropout_dict=dropout_dict,
      heads=8,
      use_attention=use_attention,
      use_summary_node=use_summary_node
    )
  elif mode == "with_topic_proxy":
    logger.info("Starting With Topic Proxy BiLSTM")
    model = TopicProxyBiLSTMGAT(
      text_dim=text_dim,
      vision_dim=vision_dim,
      audio_dim=audio_dim,
      hidden_channels=256,
      num_layers=num_layers,
      num_classes=2,
      dropout_dict=dropout_dict,
      heads=8,
      use_attention=use_attention,
      use_summary_node=use_summary_node
    )
  else:
    logger.info("Starting With BiLSTM")
    model = BiLSTMGAT(
      text_dim=text_dim,
      vision_dim=vision_dim,
      audio_dim=audio_dim,
      hidden_channels=256,
      num_layers=num_layers,
      num_classes=2,
      dropout_dict=dropout_dict,
      heads=8,
      use_attention=use_attention,
      use_summary_node=use_summary_node
    )

  if optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "MomentumSGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)


def objective(trial, config, mode): # without Attention
  lr = trial.suggest_float("lr", config['training']['lr_list'][0], config['training']['lr_list'][1])
  weight_decay = trial.suggest_float("weight_decay", config['training']['weight_decay_list'][0], config['training']['weight_decay_list'][1]) # change
  optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "MomentumSGD"])
  num_layers = trial.suggest_int("num_layers", config['model']['num_layers_list'][0], config['model']['num_layers_list'][1])
  dropout = trial.suggest_float("dropout", config['model']['dropout_list'][0], config['model']['dropout_list'][1])
  use_summary_node = trial.suggest_bool("use_summary_node")

  if optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif optimizer == "MomentumSGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)

  



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', default=100, type=int,
                      help='Number of training epochs.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='Path to latest checkpoint (default: none).')
  parser.add_argument('--config', type=str, default='search_grid.yaml',
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

  # 임시 설정
  text_dim = 300
  vision_dim = 300
  audio_dim = 300
  mode = "df"


  study = optuna.create_study(direction="maximize")

  study.optimize(
      lambda trial: bilstm_objective(trial, config, mode, text_dim, vision_dim, audio_dim),
      n_trials=50
  )