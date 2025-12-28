from sentence_transformers import SentenceTransformer
import os, sys
from .. import path_config
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .._multimodal_model_gru.GAT import GATClassifier
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..preprocessing import process_transcription, process_audio, process_vision

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

kor_to_eng_dict = {
  "심리 상태 및 감정": "Psychological State and Emotional Well-being", 
  "개인 특성 및 취미": "Personal Traits and Life Experiences",
  "생활 환경": "Living Conditions and Lifestyle Interests",
  "경력, 교육 및 군 복무": "Career, Education, and Military Service History",
  "대인 관계 및 가족": "Interpersonal Relationships and Family Dynamics"
}

def pad_sequence_numpy(seq, max_len):
  seq_len = len(seq)
  feature_dim = seq.shape[1]
  
  if seq_len >= max_len:
    return seq[:max_len, :]
  else:
    padding = np.zeros((max_len - seq_len, feature_dim))
    return np.vstack([seq, padding])

def make_graph(
    ids,
    labels,
    model_name,
    time_interval,
    colab_path=None,
    use_summary_node=True,
    t_t_connect=False,
    v_a_connect=False,
    visualization=False,
    explanation=False
  ):
  """
  make_graph's Docstring
  
  :param ids: List of patient ids
  :param labels: List of depression labels
  :param model_name: Language model name(HuggingFace)
  :param time_interval: Time interval for seperating node (unit: second)
  :param colab_path: Write your colab dataset path if you're using colab
  :param use_summary_node: Whether you want to use summary node (else, the model will do pooling with topic nodes)
  :param t_t_connect: Whether you want to connect text to text nodes regarding their temporal relationship
  :param v_a_connect: Whether you want to connect vision to audio (or audio to vision) for aligning two multimodalities
  :param visualization: Whether you want to visualize the graph construction (for a simple image, data is partially sampled)
  :param explanation: Whether you want to use GNNExplainer or analyze specifically on the model
  """
  try:
    MAX_SEQ_LEN_VISION = time_interval * 30   # 1 data for vision = 0.0333 seconds
    MAX_SEQ_LEN_AUDIO = time_interval * 40    # 1 data for audio = 0.01 seconds -> with max_pooling(kernel_size=3), 1 data = 0.3 seconds

    finish_utterance = ["asked everything", "asked_everything", "it was great chatting with you"]
    EXCLUDED_SESSIONS = ['342', '394', '398', '460']
    INTERRUPTED_SESSIONS = ['373', '444']
    NO_ALLIE = ['458', '451', '480']
    blacklist = EXCLUDED_SESSIONS + INTERRUPTED_SESSIONS + NO_ALLIE

    filtered_data = [(id, label) for id, label in zip(ids, labels) if str(id) not in blacklist]
    if len(filtered_data) < len(ids):
      logger.warning(f"Filtered out {len(ids) - len(filtered_data)} sessions from blacklist")

    filtered_ids, filtered_labels = zip(*filtered_data) if filtered_data else ([], [])

    logger.info("Getting your model")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded")
    
    graphs = []
    
    # v_scaler = StandardScaler()
    # a_scaler = StandardScaler()
    # v_scaler = RobustScaler()
    # a_scaler = RobustScaler()
    
    logger.info("Switching CSV into Graphs")
    
    if colab_path is not None:
      logger.info(f"Using Colab Path: {colab_path}")

    for graph_idx, id in tqdm(enumerate(filtered_ids), desc="Dataframe -> Graph", total=len(filtered_ids)):
      if colab_path is not None:
        df = pd.read_csv(os.path.join(colab_path, 'Transcription Topic 2', f"{id}_transcript_topic.csv"))
        v_df = pd.read_csv(os.path.join(colab_path, 'Vision Summary', f"{id}_vision_summary.csv"))
        a_df = pd.read_csv(os.path.join(colab_path, 'Audio Summary', f"{id}_audio_summary.csv"))
        
      else:
        df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Transcription Topic 2', f"{id}_transcript_topic.csv"))
        v_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Vision Summary', f"{id}_vision_summary.csv"))
        a_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Audio Summary', f"{id}_audio_summary.csv"))
    
      try:
        df.topic = df.topic.ffill()
        df = df.reset_index()
        search_pattern = '|'.join(finish_utterance)
        condition = df['value'].str.contains(search_pattern, na=False)
        terminate_index = df.index[condition]
        if not terminate_index.empty:
          df = df.iloc[:terminate_index.values[0]]

        utterances, topics, start_stop_list, start_stop_list_ellie = process_transcription(df, time_interval)

        # Vision Scaling
        vision_df = process_vision(v_df, start_stop_list_ellie)
        vision_df = vision_df.replace([np.inf, -np.inf], np.nan).fillna(0)      
        # vision_timestamps = vision_df['timestamp'].values
        # vision_df = vision_df.drop(columns=['timestamp'])
        # vision_scaled = v_scaler.fit_transform(vision_df.values)
        # vision_df = pd.DataFrame(vision_scaled, columns=vision_df.columns)
        # vision_df['timestamp'] = vision_timestamps

        # Audio Scaling
        audio_df = process_audio(a_df, start_stop_list_ellie)
        audio_df = audio_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        if audio_df.shape[1] == 0:
          logger.warning("No audio features found! Adding a dummy feature.")
          audio_df['dummy_audio'] = 0.0
        # elif audio_df.shape[1] > 0:
          # audio_values = a_scaler.fit_transform(audio_df.values)
          # audio_df = pd.DataFrame(audio_values, columns=audio_df.columns)

        # Topic/Summary nodes
        unique_topics = np.unique(np.array(topics))
        topic_node_id_dict = {
          t : idx
          for idx, t in enumerate(unique_topics)
        }

        topic_nodes = model.encode(unique_topics)
        if use_summary_node:
          summary_node = np.average(topic_nodes, axis=0).reshape(1, -1)

        transcription_list = []
        vision_seq_list = [] # Vision LSTM
        audio_seq_list = []  # Audio LSTM
        vision_lengths_list = []
        audio_lengths_list = []

        node_types = []
        if use_summary_node:
          node_types.append('summary')
        node_types.extend(['topic'] * len(topic_nodes))

        start_offset = 1 if use_summary_node else 0
        current_node_idx = start_offset + len(topic_nodes)

        source_nodes = []
        target_nodes = []

        # Topic -> Summary
        if use_summary_node:
       	  for i in range(len(topic_nodes)):
            source_nodes.append(start_offset + i)
            target_nodes.append(0)
            
        # Topic <-> Topic
        num_topics = len(topic_nodes)
        for i in range(num_topics):
          source_idx = start_offset + i
          for j in range(i + 1, num_topics):
            target_idx = start_offset + j
            source_nodes.append(source_idx)
            target_nodes.append(target_idx)
            source_nodes.append(target_idx)
            target_nodes.append(source_idx)
        
        global_prev_t_node_id = None
        # Text
        if visualization:
          utterances = utterances[::5]
          topics = topics[::5]
          start_stop_list = start_stop_list[::5]

        t_embeds = model.encode(utterances)
        for t_emb, topic, (start, stop) in zip(t_embeds, topics, start_stop_list):
          topic_node_id = topic_node_id_dict[topic]

          # Text Node
          transcription_list.append(t_emb)
          t_node_id = current_node_idx
          node_types.append('transcription')
          current_node_idx += 1

          # Text -> Topic
          source_nodes.append(t_node_id)
          target_nodes.append(start_offset + topic_node_id)

          # Text -> Text
          if global_prev_t_node_id is not None and t_t_connect:
            source_nodes.append(global_prev_t_node_id)
            target_nodes.append(t_node_id)
            source_nodes.append(t_node_id)
            target_nodes.append(global_prev_t_node_id)

          global_prev_t_node_id = t_node_id

          # Vision Node
          v_seq = vision_df.loc[(start <= vision_df['timestamp']) & (vision_df['timestamp'] <= stop)]
          v_target = v_seq.drop(columns=['timestamp']).values

          if len(v_target) > 0:
            actual_v_len = min(len(v_target), MAX_SEQ_LEN_VISION)

            v_seq_padded = pad_sequence_numpy(v_target, MAX_SEQ_LEN_VISION) # [Seq, Dim]
            vision_seq_list.append(v_seq_padded)
            vision_lengths_list.append(actual_v_len) # 길이 저장

            v_node_id = current_node_idx
            node_types.append('vision')
            current_node_idx += 1

            # Vision -> Text
            source_nodes.append(v_node_id)
            target_nodes.append(t_node_id)

          # Audio Node
          start_idx = int(start*100)
          stop_idx = int(stop*100) + 1
          a_seq = audio_df[(start_idx <= audio_df['index']) & (audio_df['index'] <= stop_idx)]
          a_target = a_seq.drop(['index'], axis=1).values

          # a_tensor = torch.FloatTensor(a_target).T.unsqueeze(0)

          # if a_tensor.shape[-1] >= 3:
          #   downsampled_a_tensor = F.avg_pool1d(a_tensor, kernel_size=3)
          # else:
          #   downsampled_a_tensor = a_tensor

          # downsampled_a_target = downsampled_a_tensor.squeeze(0).permute(1,0).cpu().numpy()

          # down sampling
          downsampled_a_target = a_target[::3]

          if len(downsampled_a_target)>0:
            actual_a_len = min(len(downsampled_a_target), MAX_SEQ_LEN_AUDIO)
          # if len(a_target)>0:
          #   actual_a_len = min(len(a_target), MAX_SEQ_LEN_AUDIO)

            a_seq_padded = pad_sequence_numpy(downsampled_a_target, MAX_SEQ_LEN_AUDIO)
            # a_seq_padded = pad_sequence_numpy(a_target, MAX_SEQ_LEN_AUDIO)
            audio_seq_list.append(a_seq_padded)
            audio_lengths_list.append(actual_a_len) # 길이 저장

            a_node_id = current_node_idx
            node_types.append('audio')
            current_node_idx += 1

            # Audio -> Text
            source_nodes.append(a_node_id)
            target_nodes.append(t_node_id)
          
          if (v_node_id and a_node_id) and v_node_id == a_node_id-1 and v_a_connect:
            source_nodes.append(v_node_id)
            target_nodes.append(a_node_id)
            source_nodes.append(a_node_id)
            target_nodes.append(v_node_id)
    
        # Static features (Summary, Topic, Text)
        feature_parts = []
        if use_summary_node:
          feature_parts.append(summary_node)
        feature_parts.append(topic_nodes)
        feature_parts.append(np.array(transcription_list))

        static_features = np.concatenate(feature_parts, axis=0)
        text_dim = static_features.shape[1]

        # total x tensor -> initiation
        total_num_nodes = len(node_types)
        x = torch.zeros((total_num_nodes, text_dim), dtype=torch.float)

        # fill static features
        text_indices = [i for i, nt in enumerate(node_types) if nt in ['summary', 'topic', 'transcription']]
        x[text_indices] = torch.tensor(static_features, dtype=torch.float)
        
        # Vision/Audio 노드를 1.0으로 초기화 (GNNExplainer를 위한 스위치 역할)
        vision_indices = [i for i, nt in enumerate(node_types) if nt == 'vision']
        audio_indices = [i for i, nt in enumerate(node_types) if nt == 'audio']

        if vision_indices:
          x[vision_indices] = 1.0
        if audio_indices:
          x[audio_indices] = 1.0

        # Vision
        vision_dim = len(vision_df.columns) - 1
        if len(vision_seq_list) > 0:
          vision_data_np = np.array(vision_seq_list)
          x_vision = torch.tensor(vision_data_np, dtype=torch.float)
          data_vision_lengths = torch.tensor(vision_lengths_list, dtype=torch.long)
        else:
          x_vision = torch.empty((0, MAX_SEQ_LEN_VISION, vision_dim))
          data_vision_lengths = torch.tensor([], dtype=torch.long)

        # Audio
        audio_dim = len(audio_df.columns) - 1
        if len(audio_seq_list) > 0:
            audio_data_np = np.array(audio_seq_list)
            # (N_audio, Seq_Len, Dim) 형태로 생성
            x_audio = torch.tensor(audio_data_np, dtype=torch.float)
            data_audio_lengths = torch.tensor(audio_lengths_list, dtype=torch.long)
        else:
            x_audio = torch.empty((0, MAX_SEQ_LEN_AUDIO, audio_dim))
            data_audio_lengths = torch.tensor([], dtype=torch.long)

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        y = torch.tensor([filtered_labels[graph_idx]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, node_types=node_types)
        data.x_vision = x_vision
        data.x_audio = x_audio
        data.vision_lengths = data_vision_lengths
        data.audio_lengths = data_audio_lengths

        graphs.append(data)

      except Exception as e:
        logger.error(f"Index: {graph_idx}, Id: {id}, Error: {e}")
        import traceback; traceback.print_exc()

    if len(graphs) > 0:
      v_dim = graphs[0].x_vision.shape[-1]
      a_dim = graphs[0].x_audio.shape[-1]
    else:
      v_dim = 0
      a_dim = 0

    if explanation:
      return graphs, (text_dim, v_dim, a_dim), topic_node_id_dict
    else:
      return graphs, (text_dim, v_dim, a_dim)
  
  except Exception as e:
    logger.error(e)
    if explanation:
      return [], (0, 0, 0), None
    else:
      return [], (0, 0, 0)

if __name__=="__main__":
  # train_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))

  # logger.info(f"Sampling Dataframes(n=8)")
  # train_df = train_df.sample(n=8)
  
  # train_id = train_df.Participant_ID.tolist()
  # train_label = train_df.PHQ8_Binary.tolist()

  train_id = [300]
  train_label = [0]

  logger.info(f"Using Sample ID: {train_id[0]}")

  logger.info(f"Total samples: {len(train_id)}")
  logger.info(f"Labels distribution: {pd.Series(train_label).value_counts().to_dict()}")
  logger.info("-" * 50)

  train_graphs, (t_dim, v_dim, a_dim) = make_graph(
    ids = train_id,
    labels = train_label,
    time_interval=10,
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    use_summary_node=True,
    v_a_connect=False,
    visualization=True)
  logger.info(f"Transcription dim: {t_dim}")
  logger.info(f"Vision dim: {v_dim}")
  logger.info(f"Audio dim: {a_dim}")
  logger.info("-" * 50)
  logger.info(f"Total graphs created: {len(train_graphs)}")
  logger.info(f"Type of first graph: {type(train_graphs[0])}")
  logger.info("-" * 50)
  
  # ============ 첫 번째 그래프 상세 확인 ============
  sample_graph = train_graphs[0]
  logger.info(f"[Sample Graph 0]")
  logger.info(f"  - Number of nodes: {sample_graph.num_nodes}")
  logger.info(f"  - Number of edges: {sample_graph.num_edges}")
  logger.info(f"  - Node feature shape: {sample_graph.x.shape}")  # [num_nodes, feature_dim]
  logger.info(f"  - Edge index shape: {sample_graph.edge_index.shape}")  # [2, num_edges]
  logger.info(f"  - Label: {sample_graph.y.item()}")
  logger.info(f"  - Has summary node: {sample_graph.num_nodes > 0}")
  logger.info("-" * 50)
  
  # ============ 여러 그래프 통계 ============
  logger.info(f"[Graph Statistics]")
  num_nodes_list = [g.num_nodes for g in train_graphs]
  num_edges_list = [g.num_edges for g in train_graphs]
  feature_dims = [g.x.shape[1] for g in train_graphs]
  
  logger.info(f"  Node counts - Min: {min(num_nodes_list)}, Max: {max(num_nodes_list)}, Avg: {sum(num_nodes_list)/len(num_nodes_list):.2f}")
  logger.info(f"  Edge counts - Min: {min(num_edges_list)}, Max: {max(num_edges_list)}, Avg: {sum(num_edges_list)/len(num_edges_list):.2f}")
  logger.info(f"  Feature dimension: {feature_dims[0]} (all same: {len(set(feature_dims)) == 1})")
  logger.info("-" * 50)
  
  # ============ 라벨 분포 확인 ============
  labels = [g.y.item() for g in train_graphs]
  logger.info(f"[Label Distribution in Graphs]")
  for label, count in pd.Series(labels).value_counts().items():
    logger.info(f"  Class {label}: {count} graphs ({count/len(labels)*100:.1f}%)")
  logger.info("-" * 50)
  
  # ============ 그래프 구조 샘플링 확인 ============
  logger.info(f"[Sample Graph Structures]")
  for i in [0, len(train_graphs)//2, -1]:  # 첫번째, 중간, 마지막
    g = train_graphs[i]
    logger.info(f"  Graph {i}: {g.num_nodes} nodes, {g.num_edges} edges, label={g.y.item()}")
  logger.info("-" * 50)
  
  # ============ 에지 연결성 확인 (첫 번째 그래프) ============
  logger.info(f"[Edge Connectivity Check - Graph 0]")
  edge_index = sample_graph.edge_index
  logger.info(f"  First 5 edges (source -> target):")
  for i in range(min(5, edge_index.shape[1])):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    logger.info(f"    {src} -> {dst}")
  logger.info("-" * 50)

  logger.info("Providing Loader/Model")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
  model = GATClassifier(
      text_dim=train_graphs[0].x.shape[1],
      vision_dim=v_dim,
      audio_dim=a_dim,
      hidden_channels=256,
      num_layers=3,
      num_classes=2,
      gru_num_layers=2,
      dropout_dict={
          'text_dropout': 0.3,
          'graph_dropout': 0.2,
          'vision_dropout': 0.4,
          'audio_dropout': 0.4
      },
      heads=8,
      use_attention=False,
      use_summary_node=True
  ).to(device)
  logger.info("Loader/Model Ready")
  logger.info("-" * 50)

  logger.info("Testing Model")
  model.eval()
  with torch.no_grad():
    try:
      for batch in tqdm(train_loader):
        batch = batch.to(device)
        out = model(batch)
      logger.info("Done Testing -> Ready to train.")
    except Exception as e:
      import traceback; traceback.print_exc()
      logger.error(e)
      logger.error("Error in Testing -> need to fix")

  logger.info("-" * 50)
  logger.info("Visualization")
  save_dir = os.path.join(path_config.ROOT_DIR, 'sample')
  os.makedirs(save_dir, exist_ok=True)
  color_map = {
    'summary': '#FF6B6B',      # 빨강
    'topic': '#4ECDC4',        # 청록
    'transcription': '#45B7D1', # 파랑
    'vision': '#FFA07A',       # 주황
    'audio': '#98D8C8'         # 민트
  }

  node_colors = [color_map.get(node_type, 'gray') for node_type in train_graphs[0].node_types]

  G = to_networkx(train_graphs[0], to_undirected=False)

  # 3. 노드 ID(0, 1, 2)에 레이블(A, B, C) 지정
  node_labels = {}
  idx = 0

  if sample_graph.node_types[0] == 'summary':
    node_labels[idx] = 'Summary'
    idx += 1
  
  for i in range(idx, sample_graph.num_nodes):
    node_labels[i] = ""

  # 4. 그래프 시각화
  plt.figure(figsize=(15, 15))
  pos = nx.spring_layout(G) # 노드 위치 결정 알고리즘 (레이아웃)

  nx.draw(G, pos,
          with_labels=True,
          labels=node_labels, # A, B, C 레이블 사용
          node_color=node_colors,
          node_size=100, 
          font_size=10,
          font_weight='bold',
          edge_color='gray',
          arrows=True, # 화살표 표시 (방향성) # type: ignore
          arrowstyle='->',
          arrowsize=20,
          font_family='Malgun Gothic',
          
        )

  legend_elements = [
      mpatches.Patch(color=color_map['summary'], label='Summary'),
      mpatches.Patch(color=color_map['topic'], label='Topic'),
      mpatches.Patch(color=color_map['transcription'], label='Transcription'),
      mpatches.Patch(color=color_map['vision'], label='Vision'),
      mpatches.Patch(color=color_map['audio'], label='Audio')
  ]
  plt.legend(handles=legend_elements, loc='upper right')

  plt.title("Graph Visualization by Node Type")
  plt.axis('off')
  plt.tight_layout()
  save_path = os.path.join(save_dir, f'{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}_graph_img.png')
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.close()
  logger.info(f"Graph visualization saved to: {save_path}")
 

  