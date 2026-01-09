from sentence_transformers import SentenceTransformer
import os, sys
from .. import path_config
import pandas as pd
import numpy as np

from tqdm import tqdm
from loguru import logger

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..graph_construct import Graph_Constructor, GraphConfig
from .._multimodal_model_gru.GAT import GATClassifier, GATJKClassifier

import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib.patches as mpatches
import networkx as nx

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

# Topic Deleted -> Text to Text connection is activated
class GRU_GC(Graph_Constructor):
  def make_graph(
      self,
      ids : list,
      labels : list,
      config : GraphConfig
  ):
    try:
      config.t_t_connect = True

      self.MAX_SEQ_LEN_VISION = config.time_interval * 30  # 1 data for vision = 0.0333 seconds
      self.MAX_SEQ_LEN_AUDIO = config.time_interval * 40  # 1 data for audio = 0.01 seconds -> with max_pooling(kernel_size=3), 1 data = 0.3 seconds

      filtered_ids, filtered_labels = self.filter_data(ids, labels)

      logger.info("Getting your model")
      language_model = SentenceTransformer(config.model_name)
      logger.info("Model loaded")

      graphs = []

      # v_scaler = StandardScaler()
      # a_scaler = StandardScaler()
      # v_scaler = RobustScaler()
      # a_scaler = RobustScaler()

      logger.info("Switching CSV into Graphs")

      if config.colab_path is not None:
        logger.info(f"Using Colab Path: {config.colab_path}")

      for graph_idx, id_ in tqdm(enumerate(filtered_ids), desc="Dataframe -> Graph", total=len(filtered_ids)):
        try:
          vision_df, audio_df, utterances, topics, start_stop_list, start_stop_list_ellie = \
            self.prepare_df(
              id_ = id_,
              time_interval = config.time_interval,
              colab_path = config.colab_path
            )

          # Static Nodes (X)
          summary_node = []
          topic_nodes = []
          transcription_list = []
          proxy_list = []

          # Non Static (Vision/Audio) Nodes
          vision_seq_list = []  # Vision GRU
          audio_seq_list = []  # Audio GRU

          # X_len
          vision_lengths_list = []
          audio_lengths_list = []

          # X_type
          node_types = []

          # edges
          source_nodes = []
          target_nodes = []

          # Previous text node for temporal connection
          global_prev_t_node_id = None

          if config.visualization:
            utterances = utterances[::8]
            logger.info(f"TOTAL NUMBER OF DATA: {len(utterances)}")
            start_stop_list = start_stop_list[::8]

          # Embedding text transcriptions
          t_embeds = language_model.encode(utterances)

          if config.use_summary_node:
            node_types.append('summary')
            summary_node = list(np.average(t_embeds, axis=0).reshape(1, -1))
            start_offset = 1
          else:
            start_offset = 0

          # Initialize node index
          current_node_idx = start_offset

          for t_emb, (start, stop) in zip(t_embeds, start_stop_list):

            # Text Nodes
            transcription_list, t_node_id, node_types, current_node_idx, source_nodes, target_nodes, global_prev_t_node_id = \
              Graph_Constructor.Text_node_wo_Topic_wo_Proxy(
                t_emb = t_emb,
                transcription_list = transcription_list,
                node_types = node_types,
                source_nodes = source_nodes,
                target_nodes = target_nodes,
                current_node_idx = current_node_idx,
                global_prev_t_node_id = global_prev_t_node_id,
                t_t_connect = config.t_t_connect,
                use_summary_node = config.use_summary_node
              )

            # Vision & Audio Nodes
            vision_seq_list, vision_lengths_list, v_node_id, current_node_idx, source_nodes, target_nodes = \
              self.V_node_T(
                vision_df=vision_df,
                vision_seq_list=vision_seq_list,
                vision_lengths_list=vision_lengths_list,
                node_types=node_types,
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                current_node_idx=current_node_idx,
                target_node_id=t_node_id,
                start=start,
                stop=stop,
              )
            audio_seq_list, audio_lengths_list, a_node_id, current_node_idx, source_nodes, target_nodes = \
              self.A_node_T(
                audio_df=audio_df,
                audio_seq_list=audio_seq_list,
                audio_lengths_list=audio_lengths_list,
                node_types=node_types,
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                current_node_idx=current_node_idx,
                target_node_id=t_node_id,
                start=start,
                stop=stop,
              )

            if (v_node_id and a_node_id) and v_node_id == a_node_id - 1 and config.v_a_connect:
              source_nodes.append(v_node_id)
              target_nodes.append(a_node_id)
              source_nodes.append(a_node_id)
              target_nodes.append(v_node_id)

          x, text_dim = Graph_Constructor.concat_features(
            transcription_list = transcription_list,
            node_types = node_types,
            summary_node = summary_node,
            topic_nodes = topic_nodes,
            proxy_list = proxy_list
          )

          vision_dim = len(vision_df.columns) - 1
          audio_dim = len(audio_df.columns) - 1

          x_vision, data_vision_lengths = \
            self.V_to_X_T(
              vision_seq_list=vision_seq_list,
              vision_lengths_list=vision_lengths_list,
              vision_dim=vision_dim
            )
          x_audio, data_audio_lengths = \
            self.A_to_X_T(
              audio_seq_list=audio_seq_list,
              audio_lengths_list=audio_lengths_list,
              audio_dim=audio_dim
            )

          edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
          y = torch.tensor([filtered_labels[graph_idx]], dtype=torch.long)

          data = Data(x=x, edge_index=edge_index, y=y, node_types=node_types)
          data.x_vision = x_vision
          data.x_audio = x_audio
          data.vision_lengths = data_vision_lengths
          data.audio_lengths = data_audio_lengths

          if config.visualization:
            logger.info(f"X_Vision: {data.x_vision.shape}")
            logger.info(f"X_Audio: {data.x_audio.shape}")
            logger.info(f"X_Vision_Len: {data.vision_lengths.shape}")
            logger.info(f"X_Vision_Len: {data.vision_lengths}")
            logger.info(f"X_Audio_Len: {data.audio_lengths.shape}")
            logger.info(f"X_Audio_Len: {data.audio_lengths}")

          graphs.append(data)

        except Exception as e:
          logger.error(f"Index: {graph_idx}, Id: {id_}, Error: {e}")
          import traceback; traceback.print_exc()

      if len(graphs) > 0:
        v_dim = graphs[0].x_vision.shape[-1]
        a_dim = graphs[0].x_audio.shape[-1]
      else:
        v_dim = 0
        a_dim = 0

      if config.explanation:
        return graphs, (text_dim, v_dim, a_dim), (utterances, vision_seq_list, audio_seq_list)
      else:
        return graphs, (text_dim, v_dim, a_dim)

    except Exception as e:
      logger.error(e)
      if config.explanation:
        return [], (0, 0, 0), (None, None, None)
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

  graph_config = GraphConfig(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    time_interval=5,
    use_summary_node=False,
    v_a_connect=False,
    visualization=True
  )
  gc = GRU_GC()

  train_graphs, (t_dim, v_dim, a_dim) = gc.make_graph(
    ids=train_id,
    labels=train_label,
    config=graph_config
  )
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

  dropout_dict = {

  }

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
 

  