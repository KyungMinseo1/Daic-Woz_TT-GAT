from sentence_transformers import SentenceTransformer
import os, path_config, sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GAT_unimodal import GATClassifier
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

def make_graph(ids, labels, model_name):
  try:
    finish_utterance = ["asked everything", "asked_everything", "it was great chatting with you"]

    logger.info("Getting your model")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded")
    dataframes = []
    logger.info("Reading CSV")

    for id in tqdm(ids, desc="Loading Dataframes"):
      df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Transcription_Topic', f"{id}_transcript_topic.csv"))
      dataframes.append(df)

    graphs = []
    dict_list = []

    logger.info("Switching CSV into Graphs")
    for graph_idx, df in tqdm(enumerate(dataframes), desc="Dataframe -> Graph", total=len(dataframes)):
      try:
        participant_dict = {
          t : []
          for t in df.topic.unique() if type(t)==str
        }
        start_stop_dict = {}

        df.topic = df.topic.ffill()
        df = df.reset_index()
        search_pattern = '|'.join(finish_utterance)
        condition = df['value'].str.contains(search_pattern, na=False)
        terminate_index = df.index[condition]
        df = df.iloc[:terminate_index.values[0]]
        participant_df = df[df.speaker=='Participant']

        previous_index = None
        previous_topic = None
        temp = ""
        start_time = 0
        stop_time = 0

        for idx, row in participant_df.iterrows():
          value = row['value']
          if pd.isna(value):
            value_str = ""
          else:
            value_str = str(value)

          # 연속 여부 판단
          if previous_index == row['index']-1 and previous_topic==row.topic:
            temp += value_str + ". "
            stop_time = row.stop_time
          else:
            if temp!="" and not pd.isna(previous_topic):
              participant_dict[previous_topic].append(temp)
              start_stop_dict[temp] = [start_time, stop_time]
            previous_topic = row.topic
            temp = value_str
            start_time = row.start_time
            stop_time = row.stop_time

          previous_index = row['index']

        # 마지막 값 저장
        if temp!="" and not pd.isna(previous_topic):
          participant_dict[previous_topic].append(temp)
          start_stop_dict[temp] = [start_time, stop_time]

        s_k_num = len(participant_dict.keys())+1
        source_nodes = np.arange(1, s_k_num)
        target_nodes = np.full(source_nodes.shape, 0)
        topic_nodes = model.encode(list(participant_dict.keys()))
        extra_nodes = None

        for idx, v in enumerate(participant_dict.values()):
          if extra_nodes is None:
            start_index = s_k_num
          else:
            start_index = s_k_num + len(extra_nodes)

          v_embed = model.encode(v)
          num_utterances = len(v_embed)
          if extra_nodes is None:
            extra_nodes = v_embed
          else:
            extra_nodes = np.concatenate([extra_nodes, v_embed])

          temp_source_nodes = np.arange(start_index, start_index+num_utterances)
          temp_target_nodes = np.full((num_utterances,), idx+1)

          if num_utterances > 1:
            t_forward_src = temp_source_nodes[:-1]
            t_forward_tgt = temp_source_nodes[1:]
            t_backward_src = temp_source_nodes[1:]
            t_backward_tgt = temp_source_nodes[:-1]
          else:
            t_forward_src = t_forward_tgt = t_backward_src = t_backward_tgt = np.array([], dtype=int)

          source_nodes = np.concatenate([source_nodes,temp_source_nodes, t_forward_src, t_backward_src])
          target_nodes = np.concatenate([target_nodes,temp_target_nodes, t_forward_tgt, t_backward_tgt])
          
        s_node = np.average(topic_nodes, axis=0).reshape(1, -1)

        if extra_nodes is None:
          print(f"#{idx} data has empty nodes -> continue to next")
          continue
        else:
          total_nodes = np.concatenate([s_node, topic_nodes, extra_nodes])
        x = torch.tensor(total_nodes, dtype=torch.float)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        y = torch.tensor([labels[graph_idx]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
        dict_list.append(participant_dict)
      except Exception as e:
        logger.error(f"Index:{graph_idx}: {e}")
        import traceback
        traceback.print_exc()
    return graphs, dict_list
  except Exception as e:
    logger.error(e)

if __name__=="__main__":
  train_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))

  logger.info(f"Sampling Dataframes(n=8)")
  train_df = train_df.sample(n=8)
  
  train_id = train_df.Participant_ID.tolist()
  train_label = train_df.PHQ8_Binary.tolist()

  logger.info(f"Total samples: {len(train_id)}")
  logger.info(f"Labels distribution: {pd.Series(train_label).value_counts().to_dict()}")
  logger.info("-" * 50)

  train_graphs, dict_list = make_graph(train_id, train_label, model_name='sentence-transformers/all-MiniLM-L6-v2')
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
      in_channels=train_graphs[0].x.shape[1],
      hidden_channels=1024,
      num_classes=2,
      heads=8,
      dropout=0.1
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
      logger.error(e)
      logger.error("Error in Testing -> need to fix")

  logger.info("-" * 50)
  logger.info("Visualization")
  save_dir = os.path.join(path_config.ROOT_DIR, 'graph', 'sample')
  os.makedirs(save_dir, exist_ok=True)

  G = to_networkx(train_graphs[0], to_undirected=False)

  # 3. 노드 ID(0, 1, 2)에 레이블(A, B, C) 지정
  node_labels = {i+1:k
                for i, k in enumerate(dict_list[0].keys())}
  node_labels[0] = 'Summary'

  # 4. 그래프 시각화
  plt.figure(figsize=(7, 5))
  pos = nx.spring_layout(G) # 노드 위치 결정 알고리즘 (레이아웃)

  nx.draw(G, pos,
          with_labels=True,
          labels=node_labels, # A, B, C 레이블 사용
          node_size=100, 
          font_size=10,
          font_weight='bold',
          edge_color='gray',
          arrows=True, # 화살표 표시 (방향성) # type: ignore
          arrowstyle='->',
          arrowsize=20,
          font_family='Malgun Gothic'
        )
  
  plt.title("Graph Visualization by Node Type")
  plt.axis('off')
  plt.tight_layout()
  save_path = os.path.join(save_dir, 'graph_sample_unimodal.png')
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.close()
  logger.info(f"Graph visualization saved to: {save_path}")
 

  