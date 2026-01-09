import os, sys
import pandas as pd
import numpy as np
from loguru import logger

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, RobustScaler

from . import path_config
from .preprocessing import process_transcription, process_audio, process_vision

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

@dataclass
class GraphConfig:
  """
  Configuration for graph construction.

  Parameters
  ----------
  model_name : str
    Language model name (HuggingFace).
  time_interval : int
    Time interval for separating nodes (unit: second).
  colab_path : Optional[str]
    Dataset path when using Google Colab.
  use_summary_node : bool
    Whether to use summary node. If False, pooling is done with topic nodes.
  t_t_connect : bool
    Whether to connect text-to-text nodes based on temporal relationships.
  v_a_connect : bool
    Whether to connect vision and audio nodes for multimodal alignment.
  visualization : bool
    Whether to visualize graph construction (partial sampling).
  explanation : bool
    Whether to use GNNExplainer or model-specific analysis.
  """
  model_name: str
  time_interval: int
  colab_path: str | None = None
  use_summary_node: bool = True
  t_t_connect: bool = False
  v_a_connect: bool = False
  visualization: bool = False
  explanation: bool = False

class Graph_Constructor(ABC):
  def __init__(self):
    self.finish_utterance = ["asked everything", "asked_everything", "it was great chatting with you"]
    self.EXCLUDED_SESSIONS = ['342', '394', '398', '460']
    self.INTERRUPTED_SESSIONS = ['373', '444']
    self.NO_ALLIE = ['458', '451', '480']
    self.blacklist = self.EXCLUDED_SESSIONS + self.INTERRUPTED_SESSIONS + self.NO_ALLIE
    self.MAX_SEQ_LEN_VISION = None
    self.MAX_SEQ_LEN_AUDIO = None

  def filter_data(self, ids, labels):
    filtered_data = [(id_, label_) for id_, label_ in zip(ids, labels) if str(id_) not in self.blacklist]
    if len(filtered_data) < len(ids):
      logger.warning(f"Filtered out {len(ids) - len(filtered_data)} sessions from blacklist")
    filtered_ids, filtered_labels = zip(*filtered_data) if filtered_data else ([], [])
    return filtered_ids, filtered_labels

  def prepare_df(
      self,
      id_,
      time_interval,
      colab_path,
      use_vision=True,
      use_audio=True
  ):
    v_df, a_df = None, None

    if colab_path is not None:
      df = pd.read_csv(os.path.join(colab_path, 'Transcription Topic 2', f"{id_}_transcript_topic.csv"))
      if use_vision:
        v_df = pd.read_csv(os.path.join(colab_path, 'Vision Summary', f"{id_}_vision_summary.csv"))
      if use_audio:
        a_df = pd.read_csv(os.path.join(colab_path, 'Audio Summary', f"{id_}_audio_summary.csv"))
    else:
      df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Transcription Topic 2', f"{id_}_transcript_topic.csv"))
      if use_vision:
        v_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Vision Summary', f"{id_}_vision_summary.csv"))
      if use_audio:
        a_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'Audio Summary', f"{id_}_audio_summary.csv"))

    try:
      df.topic = df.topic.ffill()
      df = df.reset_index()
      search_pattern = '|'.join(self.finish_utterance)
      condition = df['value'].str.contains(search_pattern, na=False)
      terminate_index = df.index[condition]
      if not terminate_index.empty:
        df = df.iloc[:terminate_index.values[0]]

      utterances, topics, start_stop_list, start_stop_list_ellie = process_transcription(df, time_interval)

      if v_df is not None:
        v_df = process_vision(v_df, start_stop_list_ellie)
        v_df = v_df.replace([np.inf, -np.inf], np.nan).fillna(0)

      if a_df is not None:
        a_df = process_audio(a_df, start_stop_list_ellie)
        a_df = a_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        if a_df.shape[1] == 0:
          logger.warning("No audio features found! Adding a dummy feature.")
          a_df['dummy_audio'] = 0.0

      return v_df, a_df, utterances, topics, start_stop_list, start_stop_list_ellie

    except Exception as e:
      logger.error(f"Error processing dataframes for id-{id_}: {e}")
      return None, None

  def V_node_T(
      self,
      vision_df: pd.DataFrame,
      vision_seq_list: list,
      vision_lengths_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      current_node_idx: int,
      target_node_id: int,
      start: float,
      stop: float
  ):
    """
    Construct Vision & Audio node **Temporal** / connection

    Parameters
    ----------
    vision_df : pd.DataFrame
      Dataframe of vision data
    vision_seq_list : list
      List of vision data in current sequence
    vision_lengths_list : list
      List of vision data lengths in current sequence (No padding regareded)
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    current_node_idx : int
      Current node index (start from 0)
    target_node_id : int
      Target node id of vision/audio nodes (can be text or proxy node)
    start : float
      Start time of a sequence
    stop : float
      End time of a sequence
    """
    assert self.MAX_SEQ_LEN_VISION, logger.error(
      "Max sequence not defined(check your MAX_SEQ_LEN_VISION)")
    # Vision Node
    v_seq = vision_df.loc[(start <= vision_df['timestamp']) & (vision_df['timestamp'] <= stop)]
    v_target = v_seq.drop(columns=['timestamp']).values

    if len(v_target) > 0:
      actual_v_len = min(len(v_target), self.MAX_SEQ_LEN_VISION)

      v_seq_padded = Graph_Constructor.pad_sequence_numpy(v_target, self.MAX_SEQ_LEN_VISION)  # [Seq, Dim]
      vision_seq_list.append(v_seq_padded)
      vision_lengths_list.append(actual_v_len)  # 길이 저장

      v_node_id = current_node_idx
      node_types.append('vision')
      current_node_idx += 1

      # Vision -> Proxy
      source_nodes.append(v_node_id)
      target_nodes.append(target_node_id)
    else:
      v_node_id = None

    return vision_seq_list, vision_lengths_list, v_node_id, current_node_idx, source_nodes, target_nodes

  def A_node_T(
      self,
      audio_df: pd.DataFrame,
      audio_seq_list: list,
      audio_lengths_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      current_node_idx: int,
      target_node_id: int,
      start: float,
      stop: float
  ):
    """
    Construct Vision & Audio node **Temporal** / connection

    Parameters
    ----------
    audio_df : pd.DataFrame
      Dataframe of audio data
    audio_seq_list : list
      List of audio data in current sequence
    audio_lengths_list : list
      List of vision data lengths in current sequence (No padding regareded)
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    current_node_idx : int
      Current node index (start from 0)
    target_node_id : int
      Target node id of vision/audio nodes (can be text or proxy node)
    start : float
      Start time of a sequence
    stop : float
      End time of a sequence
    """
    assert self.MAX_SEQ_LEN_AUDIO, logger.error(
      "Max sequence not defined(check your MAX_SEQ_LEN_AUDIO)")
    # Audio Node
    start_idx = int(start * 100)
    stop_idx = int(stop * 100) + 1
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

    if len(downsampled_a_target) > 0:
      actual_a_len = min(len(downsampled_a_target), self.MAX_SEQ_LEN_AUDIO)
      # if len(a_target)>0:
      #   actual_a_len = min(len(a_target), MAX_SEQ_LEN_AUDIO)

      a_seq_padded = Graph_Constructor.pad_sequence_numpy(downsampled_a_target, self.MAX_SEQ_LEN_AUDIO)
      # a_seq_padded = pad_sequence_numpy(a_target, MAX_SEQ_LEN_AUDIO)
      audio_seq_list.append(a_seq_padded)
      audio_lengths_list.append(actual_a_len)  # 길이 저장

      a_node_id = current_node_idx
      node_types.append('audio')
      current_node_idx += 1

      # Audio -> Proxy
      source_nodes.append(a_node_id)
      target_nodes.append(target_node_id)
    else:
      a_node_id = None

    return audio_seq_list, audio_lengths_list, a_node_id, current_node_idx, source_nodes, target_nodes

  def V_to_X_T(
      self,
      vision_seq_list: list,
      vision_lengths_list: list,
      vision_dim: int,
  ):
    """
    Vision data to X **temporal**

    Parameters
    ----------
    vision_seq_list : list
      List of vision data in current sequence
    vision_lengths_list : list
      List of vision data lengths in current sequence (No padding regareded)
    vision_dim : int
      Dimension of vision data
    """
    if len(vision_seq_list) > 0:
      vision_data_np = np.array(vision_seq_list)
      x_vision = torch.tensor(vision_data_np, dtype=torch.float)
      data_vision_lengths = torch.tensor(vision_lengths_list, dtype=torch.long)
    else:
      x_vision = torch.empty((0, self.MAX_SEQ_LEN_VISION, vision_dim))
      data_vision_lengths = torch.tensor([], dtype=torch.long)

    return x_vision, data_vision_lengths

  def A_to_X_T(
      self,
      audio_seq_list: list,
      audio_lengths_list: list,
      audio_dim: int
  ):
    """
    Audio data to X **temporal**

    Parameters
    ----------
    audio_seq_list : list
      List of audio data in current sequence
    audio_lengths_list : list
      List of audio data lengths in current sequence (No padding regareded)
    audio_dim : int
      Dimension of audio data
    """
    if len(audio_seq_list) > 0:
      audio_data_np = np.array(audio_seq_list)
      # (N_audio, Seq_Len, Dim) 형태로 생성
      x_audio = torch.tensor(audio_data_np, dtype=torch.float)
      data_audio_lengths = torch.tensor(audio_lengths_list, dtype=torch.long)
    else:
      x_audio = torch.empty((0, self.MAX_SEQ_LEN_AUDIO, audio_dim))
      data_audio_lengths = torch.tensor([], dtype=torch.long)

    return x_audio, data_audio_lengths

  @staticmethod
  def V_node_P(
      vision_df: pd.DataFrame,
      vision_pooled_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      current_node_idx: int,
      target_node_id: int,
      start: float,
      stop: float
  ):
    """
    Construct Vision & Audio node **Average Pooling** / connection

    Parameters
    ----------
    vision_df : pd.DataFrame
      Dataframe of vision data
    vision_pooled_list : list
      List of averaged pooled vision data in current sequence
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    current_node_idx : int
      Current node index (start from 0)
    target_node_id : int
      Target node id of vision/audio nodes (can be text or proxy node)
    start : float
      Start time of a sequence
    stop : float
      End time of a sequence
    """
    # Vision Node
    v_seq = vision_df.loc[(start <= vision_df['timestamp']) & (vision_df['timestamp'] <= stop)]
    v_target = v_seq.drop(columns=['timestamp']).values

    if len(v_target) > 0:
      v_pooled = np.mean(v_target, axis=0)  # [Dim,]
      vision_pooled_list.append(v_pooled)

      v_node_id = current_node_idx
      node_types.append('vision')
      current_node_idx += 1

      # Vision -> Target
      source_nodes.append(v_node_id)
      target_nodes.append(target_node_id)
    else:
      v_node_id = None

    return vision_pooled_list, v_node_id, current_node_idx, source_nodes, target_nodes

  @staticmethod
  def A_node_P(
      audio_df: pd.DataFrame,
      audio_pooled_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      current_node_idx: int,
      target_node_id: int,
      start: float,
      stop: float
  ):
    """
    Construct Vision & Audio node **Temporal** / connection

    Parameters
    ----------
    audio_df : pd.DataFrame
      Dataframe of audio data
    audio_pooled_list : list
      List of average pooled audio data in current sequence
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    current_node_idx : int
      Current node index (start from 0)
    target_node_id : int
      Target node id of vision/audio nodes (can be text or proxy node)
    start : float
      Start time of a sequence
    stop : float
      End time of a sequence
    """
    start_idx = int(start * 100)
    stop_idx = int(stop * 100) + 1
    a_seq = audio_df[(start_idx <= audio_df['index']) & (audio_df['index'] <= stop_idx)]
    a_target = a_seq.drop(['index'], axis=1).values

    if len(a_target) > 0:
      a_pooled = np.mean(a_target, axis=0)  # [Dim,]
      audio_pooled_list.append(a_pooled)

      a_node_id = current_node_idx
      node_types.append('audio')
      current_node_idx += 1

      # Audio -> Target
      source_nodes.append(a_node_id)
      target_nodes.append(target_node_id)
    else:
      a_node_id = None

    return audio_pooled_list, a_node_id, current_node_idx, source_nodes, target_nodes

  @staticmethod
  def V_to_X_P(
      vision_pooled_list: list,
      vision_dim: int,
  ):
    """
    Vision data to X **average pooling**

    Parameters
    ----------
    vision_pooled_list : list
      List of average pooled vision data in current sequence
    vision_dim : int
      Dimension of vision data
    """
    if len(vision_pooled_list) > 0:
      x_vision = torch.tensor(np.array(vision_pooled_list), dtype=torch.float)
    else:
      x_vision = torch.empty((0, vision_dim))   # Dummy dim
    return x_vision

  @staticmethod
  def A_to_X_P(
      audio_pooled_list: list,
      audio_dim: int,
  ):
    """
    Vision data to X **average pooling**

    Parameters
    ----------
    audio_pooled_list : list
      List of average pooled vision data in current sequence
    audio_dim : int
      Dimension of vision data
    """
    if len(audio_pooled_list) > 0:
      x_audio = torch.tensor(np.array(audio_pooled_list), dtype=torch.float)
    else:
      x_audio = torch.empty((0, audio_dim))     # Dummy data
    return x_audio

  @staticmethod
  def ToSu_node(
      topics : list,
      node_types : list,
      current_node_idx : int,
      source_nodes : list,
      target_nodes : list,
      language_model,
      use_summary_node):
    """
    Construct Topic & Summary node / connection

    Parameters
    ----------
    topics : list
      List of topics
    node_types : list
      List of node types
    current_node_idx : int
      Current node index (start from 0)
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    language_model
      Language model to encode topics
    use_summary_node
      Whether to use summary node
    """
    unique_topics = np.unique(np.array(topics))
    topic_node_id_dict = {
      t: idx
      for idx, t in enumerate(unique_topics)
    }

    topic_nodes = language_model.encode(unique_topics)

    if use_summary_node:
      node_types.append('summary')
    node_types.extend(['topic'] * len(topic_nodes))

    start_offset = 1 if use_summary_node else 0
    current_node_idx += start_offset + len(topic_nodes)

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

    return topic_nodes, node_types, current_node_idx, source_nodes, target_nodes, topic_node_id_dict, start_offset

  @staticmethod
  def Text_node_wo_Topic_wo_Proxy(
      t_emb,
      transcription_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      current_node_idx: int,
      global_prev_t_node_id,
      t_t_connect: bool,
      use_summary_node: bool,
  ):
    """
    Construct Text(Transcription) node / connection **without topic**

    Parameters
    ----------
    t_emb
      Embeddings of current transcription from a language model
    transcription_list : list
      List of text(transcriptions) nodes
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    current_node_idx : int
      Current node index (start from 0)
    global_prev_t_node_id
      Previous text(transcription) node id
    t_t_connect : bool
      Whether to connect text-to-text nodes based on temporal relationships.
    use_summary_node : bool
      Whether to use summary node. If False, pooling is done with topic nodes.
    """
    transcription_list.append(t_emb)
    t_node_id = current_node_idx
    node_types.append('transcription')
    current_node_idx += 1

    # Text -> Summary
    if use_summary_node:
      source_nodes.append(t_node_id)
      target_nodes.append(0)

    # Text -> Text
    if global_prev_t_node_id is not None and t_t_connect:
      source_nodes.append(global_prev_t_node_id)
      target_nodes.append(t_node_id)
      source_nodes.append(t_node_id)
      target_nodes.append(global_prev_t_node_id)

    global_prev_t_node_id = t_node_id

    return transcription_list, t_node_id, node_types, current_node_idx, source_nodes, target_nodes, global_prev_t_node_id

  @staticmethod
  def Text_node_wo_Topic(
      t_emb,
      transcription_list: list,
      proxy_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      current_node_idx: int,
      global_prev_t_node_id,
      t_t_connect: bool,
      use_summary_node: bool
  ):
    """
    Construct Text(Transcription) node / connection **without topic**

    Parameters
    ----------
    t_emb
      Embeddings of current transcription from a language model
    transcription_list : list
      List of text(transcriptions) nodes
    proxy_list : list
      List of proxy nodes
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    current_node_idx : int
      Current node index (start from 0)
    global_prev_t_node_id
      Previous text(transcription) node id
    t_t_connect : bool
      Whether to connect text-to-text nodes based on temporal relationships.
    use_summary_node : bool
      Whether to use summary node. If False, pooling is done with topic nodes.
    """
    transcription_list.append(t_emb)
    t_node_id = current_node_idx
    node_types.append('transcription')
    current_node_idx += 1

    # Proxy Node
    proxy_list.append(t_emb)
    p_node_id = current_node_idx
    node_types.append('proxy')
    current_node_idx += 1

    # Text -> Summary
    if use_summary_node:
      source_nodes.append(t_node_id)
      target_nodes.append(0)

    # Proxy -> Text
    source_nodes.append(p_node_id)
    target_nodes.append(t_node_id)

    # Text -> Text
    if global_prev_t_node_id is not None and t_t_connect:
      source_nodes.append(global_prev_t_node_id)
      target_nodes.append(t_node_id)
      source_nodes.append(t_node_id)
      target_nodes.append(global_prev_t_node_id)

    global_prev_t_node_id = t_node_id

    return transcription_list, p_node_id, node_types, current_node_idx, source_nodes, target_nodes, global_prev_t_node_id

  @staticmethod
  def Text_node_wo_Proxy(
      t_emb,
      topic: str,
      transcription_list: list,
      node_types: list,
      source_nodes: list,
      target_nodes: list,
      topic_node_id_dict: dict,
      current_node_idx: int,
      global_prev_t_node_id,
      t_t_connect: bool,
      start_offset: int
  ):
    """
    Construct Text(Transcription) node / connection **without proxy**

    Parameters
    ----------
    t_emb
      Embeddings of current transcription from a language model
    topic : str
      String type topic description of current transcription
    transcription_list : list
      List of text(transcriptions) nodes
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    topic_node_id_dict : dict
      Dictionary of topics and ids
    current_node_idx : int
      Current node index (start from 0)
    global_prev_t_node_id
      Previous text(transcription) node id
    t_t_connect : bool
      Whether to connect text-to-text nodes based on temporal relationships.
    start_offset : int
      Start location of topic nodes (if there is a summary node, given offset is 1)
    """
    topic_node_id = topic_node_id_dict[topic]
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

    return transcription_list, t_node_id, node_types, current_node_idx, source_nodes, target_nodes, global_prev_t_node_id

  @staticmethod
  def Text_node(
      t_emb,
      topic : str,
      transcription_list : list,
      proxy_list : list,
      node_types : list,
      source_nodes : list,
      target_nodes : list,
      topic_node_id_dict : dict,
      current_node_idx : int,
      global_prev_t_node_id,
      t_t_connect : bool,
      start_offset : int
  ):
    """
    Construct Text(Transcription) node / connection

    Parameters
    ----------
    t_emb
      Embeddings of current transcription from a language model
    topic : str
      String type topic description of current transcription
    transcription_list : list
      List of text(transcriptions) nodes
    proxy_list : list
      List of proxy nodes
    node_types : list
      List of node types
    source_nodes : list
      List of source nodes (source -> target)
    target_nodes : list
      List of target nodes (source -> target)
    topic_node_id_dict : dict
      Dictionary of topics and ids
    current_node_idx : int
      Current node index (start from 0)
    global_prev_t_node_id
      Previous text(transcription) node id
    t_t_connect : bool
      Whether to connect text-to-text nodes based on temporal relationships.
    start_offset : int
      Start location of topic nodes (if there is a summary node, given offset is 1)
    """
    # Text Node
    topic_node_id = topic_node_id_dict[topic]
    transcription_list.append(t_emb)
    t_node_id = current_node_idx
    node_types.append('transcription')
    current_node_idx += 1

    # Proxy Node
    proxy_list.append(t_emb)
    p_node_id = current_node_idx
    node_types.append('proxy')
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

    # Proxy -> Text
    source_nodes.append(p_node_id)
    target_nodes.append(t_node_id)

    global_prev_t_node_id = t_node_id

    return transcription_list, proxy_list, p_node_id, node_types, current_node_idx, source_nodes, target_nodes, global_prev_t_node_id

  @staticmethod
  def concat_features(
      transcription_list: list,
      node_types: list,
      summary_node: list = None,
      topic_nodes: list = None,
      proxy_list: list = None
  ):
    """
    Concatenate static features and make X for geometric data.

    Parameters
    ----------
    transcription_list : list
      List of text(transcriptions) nodes
    node_types : list
      List of node types
    summary_node : list
      List of a summary node
    topic_nodes : list
      List of topic nodes
    proxy_list : list
      List of proxy nodes
    """
    feature_parts = []
    if len(summary_node)>0:
      feature_parts.append(summary_node)
    if len(topic_nodes)>0:
      feature_parts.append(topic_nodes)
    feature_parts.append(np.array(transcription_list))
    if len(proxy_list)>0:
      feature_parts.append(np.array(proxy_list))

    static_features = np.concatenate(feature_parts, axis=0)
    text_dim = static_features.shape[1]

    # total x tensor -> initiation
    total_num_nodes = len(node_types)
    x = torch.zeros((total_num_nodes, text_dim), dtype=torch.float)

    # fill static features
    text_indices = [i for i, nt in enumerate(node_types) if nt not in ['vision', 'audio']]
    x[text_indices] = torch.tensor(static_features, dtype=torch.float)

    # Vision/Audio 노드를 1.0으로 초기화 (GNNExplainer를 위한 스위치 역할)
    vision_indices = [i for i, nt in enumerate(node_types) if nt == 'vision']
    audio_indices = [i for i, nt in enumerate(node_types) if nt == 'audio']

    if vision_indices:
      x[vision_indices] = 1.0
    if audio_indices:
      x[audio_indices] = 1.0

    return x, text_dim

  @staticmethod
  def pad_sequence_numpy(seq, max_len):
    feature_dim = seq.shape[1]

    if len(seq) >= max_len:
      return seq[:max_len, :]
    else:
      padding = np.zeros((max_len - len(seq), feature_dim))
      return np.vstack([seq, padding])

  @abstractmethod
  def make_graph(
      self,
      ids: list,
      labels: list,
      config: GraphConfig
  ):
    """
    Construct a graph from multimodal patient data.

    Parameters
    ----------
    ids : list
      List of patient IDs.
    labels : list
      List of depression labels.
    config : GraphConfig
      Graph construction configuration.
    """
    pass



