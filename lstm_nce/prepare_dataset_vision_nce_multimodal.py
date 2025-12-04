import os, config_path, sys
from loguru import logger
import pandas as pd
from multiprocessing import Manager
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk.corpus import stopwords

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

logger.info("Loading GLOVE")
glove_kv_path = os.path.join(config_path.MODEL_DIR, "glove_model.kv")

assert os.path.exists(glove_kv_path), "No GLOVE Model"

# Defining GLOVE
try:
  glove_model = KeyedVectors.load(glove_kv_path)
  logger.info("Loaded GLOVE")
except Exception as e:
  logger.error(f"Problem with your GLOVE: {e}")

# Defining STOPWORDS
nltk.download('stopwords')
stop_words_list = stopwords.words('english')
logger.info("Loaded Stopwords")

def process_transcription(df):
  finish_utterance = ["asked everything", "asked_everything", "it was great chatting with you"]

  search_pattern = '|'.join(finish_utterance)
  condition = df['value'].str.contains(search_pattern, na=False)
  terminate_index = df.index[condition]
  n_df = df.iloc[:terminate_index.values[0]]

  is_not_ellie = n_df['speaker'] != 'Ellie'
  new_group_start = (is_not_ellie) & (~is_not_ellie.shift(1, fill_value=False))
  group_id = new_group_start.cumsum()
  n_df['count'] = group_id.where(is_not_ellie, None)
  n_df.loc[~is_not_ellie, 'count'] = None
  participant_df = n_df[~pd.isna(n_df['count'])]
  group_df = participant_df.dropna(subset=['count']).groupby('count').agg(
      start_time=('start_time', 'min'), # 그룹의 가장 이른 시작 시간
      stop_time=('stop_time', 'max')   # 그룹의 가장 늦은 종료 시간
  ).reset_index()
  return participant_df, group_df

def process_vision(df):
  timestamp = df.timestamp
  ft_x = df.filter(like='ftx')
  ft_y = df.filter(like='fty')
  ft_3d_x = df.filter(like='ft_3dX')
  ft_3d_y = df.filter(like='ft_3dY')
  ft_3d_z = df.filter(like='ft_3dZ')
  au_r = df.filter(like='au').filter(like='_r')
  gz_df = df.filter(like='gz')
  gz_h = gz_df.filter(like='h')
  ps_t = df.filter(like='ps').filter(like='T')
  ps_r = df.filter(like='ps').filter(like='R')
  vision = pd.concat([timestamp, ft_x, ft_y, ft_3d_x, ft_3d_y, ft_3d_z, au_r, gz_h, ps_t, ps_r], axis=1)
  return vision

def vectorize_t_and_read_v(id, dataset, dataset2):
  try:
    TRANSCRIPTION = []
    VISION = []

    t_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'Transcription', f'{id}_transcript.csv'))
    v_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'Vision Summary', f'{id}_vision_summary.csv'))

    participant_df, group_df = process_transcription(t_df)
    vision = process_vision(v_df)

    for count in group_df['count'].tolist():
      word_list = " ".join(participant_df.loc[participant_df['count']==count].value.tolist()).split()
      for word in word_list:
        try:
          word = word.strip()
          if "'" in word:
            word = word.split("'")[0]
          if '<' in word or '>' in word or '_' in word or '[' in word or ']' in word or word in stop_words_list:
            pass
          elif word == "":
            pass
          else:
            TRANSCRIPTION.append(glove_model[word]) # final -> (word_num, 300)
        except KeyError:
          logger.error(f"{word} is not in the model.")

    for _, row in group_df.iterrows():
      start = row.start_time
      stop = row.stop_time
      v_target = vision.loc[(start <= vision.timestamp) & (vision.timestamp <= stop)]
      v_target = v_target.drop(columns=['timestamp'])
      # (seq_len, v_columns)
      v_target_list = v_target.values.tolist()

      if len(v_target_list) > 0:
        VISION.append(v_target_list)
      else:
        print(f"No Data in vision # {id}")

    # TRANSCRIPTION: (B, seq_len, 300)
    dataset.extend(TRANSCRIPTION)

    # VISION: (B, seq_len, v_columns)
    dataset2.extend(VISION)

  except Exception as e:
    print("오류 발생:",e)

if __name__=="__main__":
  mgr = Manager()
  train_transcription_dataset = mgr.list()
  train_vision_dataset = mgr.list()
  val_transcription_dataset = mgr.list()
  val_vision_dataset = mgr.list()
  
  train_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
  val_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'dev_split_Depression_AVEC2017.csv'))
  
  train_id = train_df.Participant_ID.tolist()
  val_id = val_df.Participant_ID.tolist()

  with Pool(processes=10) as p:
    with tqdm(total=len(train_id)) as pbar:
      for v in p.imap_unordered(partial(vectorize_t_and_read_v, dataset=train_transcription_dataset, dataset2=train_vision_dataset), train_id):
        pbar.update()

    with tqdm(total=len(val_id)) as pbar:
      for v in p.imap_unordered(partial(vectorize_t_and_read_v, dataset=val_transcription_dataset, dataset2=val_vision_dataset), val_id):
        pbar.update()
  print('학습 데이터 길이:',len(train_transcription_dataset))
  print('검증 데이터 길이:',len(val_transcription_dataset))
  print('input_dim:', len(train_transcription_dataset[0][0]))