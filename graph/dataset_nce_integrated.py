import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
import path_config

import torch
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from lstm_nce_multimodal import NCEMultimodalModel
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

nltk.download('stopwords', quiet=True)
stop_words_list = stopwords.words('english')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

def process_transcription_with_topic(df):
  finish_utterance = ["asked everything", "asked_everything", "it was great chatting with you"]
  search_pattern = '|'.join(finish_utterance)
  
  condition = df['value'].str.contains(search_pattern, na=False)
  terminate_index = df.index[condition]
  if terminate_index.empty:
    terminate_value = len(df)
  else:
    terminate_value = terminate_index.values[0]
  
  n_df = df.iloc[:terminate_value].copy()
  
  n_df['topic'] = n_df['topic'].ffill()

  is_not_ellie = n_df['speaker'] != 'Ellie'
  new_group_start = (is_not_ellie) & (~is_not_ellie.shift(1, fill_value=False))
  group_id = new_group_start.cumsum()

  n_df['count'] = group_id.where(is_not_ellie, pd.NA)
  participant_df = n_df.dropna(subset=['count'])
  
  # 그룹별 정보 (Start, Stop, Topic)
  group_df = participant_df.groupby('count').agg(
      start_time=('start_time', 'min'),
      stop_time=('stop_time', 'max'),
      topic=('topic', 'first') 
  ).reset_index()
  
  return participant_df, group_df

def process_vision(df):
  timestamp = df.timestamp
  ft_x = df.filter(like='ftx')
  ft_y = df.filter(like='fty')
  au_r = df.filter(like='au').filter(like='_r')
  gz_df = df.filter(like='gz')
  gz_h = gz_df.filter(like='h')
  ps_t = df.filter(like='ps').filter(like='T')
  ps_r = df.filter(like='ps').filter(like='R')
  
  vision = pd.concat([timestamp, ft_x, ft_y, au_r, gz_h, ps_t, ps_r], axis=1)
  return vision

def make_graph(ids, labels, checkpoints_dir, checkpoints_dir_):
  try:
    logger.info("Loading GloVe Model...")
    glove_model = KeyedVectors.load(os.path.join(path_config.MODEL_DIR, 'glove_model.kv'))
    
    logger.info("Loading NCE Model...")
    checkpoint_path = os.path.join(checkpoints_dir, checkpoints_dir_, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    
    best_nce_model = NCEMultimodalModel(
      x_input_dim=300,   # glove dim
      x2_input_dim=162,  # vision dim
      hidden_dim=checkpoint['config']['model']['h_dim'],
      output_dim=checkpoint['config']['model']['o_dim'],
      num_layers=checkpoint['config']['model']['depth'],
      dropout=checkpoint['config']['model']['dropout']
    )
    best_nce_model.load_state_dict(checkpoint['model_state_dict'])
    
    lstm_transcription = best_nce_model.encoder_x_k
    lstm_vision = best_nce_model.encoder_x2_k
    
    lstm_transcription.eval()
    lstm_vision.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_transcription.to(device)
    lstm_vision.to(device)

    graphs = []
    
    logger.info(f"Processing {len(ids)} participants...")

    for idx, (id, label) in enumerate(zip(tqdm(ids, desc="Graph Construction"), labels)):
      try:
        t_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Transcription_Topic', f"{id}_transcript_topic.csv"))
        v_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Vision Summary', f"{id}_vision_summary.csv"))
        
        participant_df, group_df = process_transcription_with_topic(t_df)
        vision_full = process_vision(v_df)
        vision_full = vision_full.fillna(0)
      
        batch_text_seqs = []
        batch_vision_seqs = []
        group_topics_list = []
        
        unique_topics_ordered = []
        seen_topics = set()

        for count in group_df['count'].tolist():
          word_list = " ".join(participant_df.loc[participant_df['count']==count].value.tolist()).split()
          group_trans_list = []
          for word in word_list:
            try:
              word = word.strip().split("'")[0]
              if ('<' in word) or ('>' in word) or ('[' in word) or (']' in word) or (word in stop_words_list) or (word == ""):
                pass
              else:
                group_trans_list.append(glove_model[word].tolist())
            except KeyError:
              pass
          
          row = group_df[group_df['count']==count].iloc[0]
          v_target = vision_full.loc[(row.start_time <= vision_full.timestamp) & (vision_full.timestamp <= row.stop_time)]
          v_target_list = v_target.drop(columns=['timestamp']).values.tolist()

          if len(group_trans_list) > 0 and len(v_target_list) > 0:
            batch_text_seqs.append(torch.tensor(group_trans_list, dtype=torch.float))
            batch_vision_seqs.append(torch.tensor(v_target_list, dtype=torch.float))
            
            current_topic = str(row.topic)
            group_topics_list.append(current_topic)
            
            if current_topic not in seen_topics:
              unique_topics_ordered.append(current_topic)
              seen_topics.add(current_topic)

        if len(batch_text_seqs) == 0:
          logger.warning(f"ID {id} has no valid groups. Skipping.")
          continue

        # LSTM Encoding
        padded_x = rnn_utils.pad_sequence(batch_text_seqs, batch_first=True).to(device)
        x_lens = torch.tensor([len(seq) for seq in batch_text_seqs])
        
        padded_x2 = rnn_utils.pad_sequence(batch_vision_seqs, batch_first=True).to(device)
        x2_lens = torch.tensor([len(seq) for seq in batch_vision_seqs])
        
        with torch.no_grad():
          t_feat, _ = lstm_transcription(padded_x, x_lens)   # Text Nodes
          v_feat, _ = lstm_vision(padded_x2, x2_lens)      # Vision Nodes
        
        t_feat = t_feat.cpu()
        v_feat = v_feat.cpu()
        
        topic_features = []
        topic_to_idx_map = {t: i for i, t in enumerate(unique_topics_ordered)}
        
        for t in unique_topics_ordered:
          indices = [i for i, gt in enumerate(group_topics_list) if gt == t]
          if len(indices) > 0:
            t_mean = torch.mean(t_feat[indices], dim=0)
            topic_features.append(t_mean)
          else:
            topic_features.append(torch.zeros(t_feat.size(1)))
            
        topic_features = torch.stack(topic_features) # (Num_Topics, Feature_Dim)

        summary_node = torch.mean(topic_features, dim=0, keepdim=True)

        x = torch.cat([summary_node, topic_features, t_feat, v_feat], dim=0)

        # Edge Construction
        num_topics = len(unique_topics_ordered)
        num_groups = len(t_feat)
        
        # Indices Offsets
        summary_idx = 0
        topic_start = 1
        text_start = 1 + num_topics
        vision_start = 1 + num_topics + num_groups

        src_nodes = []
        dst_nodes = []

        # Vision -> Text
        for i in range(num_groups):
          v_node = vision_start + i
          t_node = text_start + i
          src_nodes.append(v_node); dst_nodes.append(t_node)

        # Text <-> Text
        for i in range(num_groups - 1):
          curr_t = text_start + i
          next_t = text_start + i + 1
          src_nodes.append(curr_t); dst_nodes.append(next_t)
          src_nodes.append(next_t); dst_nodes.append(curr_t)

        # Text -> Topic
        for i in range(num_groups):
          t_node = text_start + i
          topic_str = group_topics_list[i]
          t_idx = topic_to_idx_map[topic_str]
          topic_node = topic_start + t_idx
          src_nodes.append(t_node); dst_nodes.append(topic_node)

        # Topic <-> Topic
        for i in range(num_topics - 1):
          curr_topic = topic_start + i
          next_topic = topic_start + i + 1
          src_nodes.append(curr_topic); dst_nodes.append(next_topic)
          src_nodes.append(next_topic); dst_nodes.append(curr_topic)

        # Topic -> Summary
        for i in range(num_topics):
          curr_topic = topic_start + i
          src_nodes.append(curr_topic); dst_nodes.append(summary_idx)

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        
        # Node Types List
        node_types = ['summary'] + \
                     ['topic'] * num_topics + \
                     ['text'] * num_groups + \
                     ['vision'] * num_groups
        
        data = Data(x=x, edge_index=edge_index, y=y, node_types=node_types)
        graphs.append(data)

      except Exception as e:
        logger.error(f"Error processing ID {id}: {e}")
        continue
    
    return graphs
      
  except Exception as e:
    logger.error(f"Fatal Error: {e}")
    return [], 0

if __name__=="__main__":
  train_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
  
  logger.info(f"Sampling Dataframes (n=8)")
  train_df = train_df.sample(n=8)
  
  train_id = train_df.Participant_ID.tolist()
  train_label = train_df.PHQ8_Binary.tolist()
  
  train_graphs, v_dim = make_graph(
    train_id, 
    train_label, 
    checkpoints_dir='checkpoints_nce', 
    checkpoints_dir_='multimodal_nce_4'
  )
  
  if len(train_graphs) > 0:
    logger.info(f"Total graphs created: {len(train_graphs)}")
    sample_graph = train_graphs[0]
    
    logger.info("Visualizing Sample Graph...")
    save_dir = os.path.join(path_config.ROOT_DIR, 'graph', 'sample')
    os.makedirs(save_dir, exist_ok=True)
    
    #  - 시각화
    color_map = {
      'summary': '#FF6B6B',  # Red
      'topic': '#FFD93D',    # Yellow
      'text': '#6BCB77',     # Green
      'vision': '#4D96FF'    # Blue
    }
    
    node_colors = [color_map.get(nt, 'gray') for nt in sample_graph.node_types]
    
    G = to_networkx(sample_graph, to_undirected=False)
    
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 12))
    
    # 라벨 약어
    labels = {}
    for i, nt in enumerate(sample_graph.node_types):
      if i == 0: labels[i] = "SUM"
      elif nt == 'topic': labels[i] = "TP"
      elif nt == 'text': labels[i] = ""
      elif nt == 'vision': labels[i] = ""

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=8, alpha=0.4)
    
    legend_elements = [
        mpatches.Patch(color=color_map['summary'], label='Summary'),
        mpatches.Patch(color=color_map['topic'], label='Topic'),
        mpatches.Patch(color=color_map['text'], label='Text'),
        mpatches.Patch(color=color_map['vision'], label='Vision')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Hierarchical Graph: Vision -> Text -> Topic -> Summary")
    plt.axis('off')
    
    save_path = os.path.join(save_dir, 'graph_hierarchical_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to {save_path}")
      
  else:
    logger.error("No graphs were created.")