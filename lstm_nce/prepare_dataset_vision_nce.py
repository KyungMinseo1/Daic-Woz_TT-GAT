import os, config_path
import pandas as pd
from multiprocessing import Manager
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial

def process_transcription(df):
  is_not_ellie = df['speaker'] != 'Ellie'
  new_group_start = (is_not_ellie) & (~is_not_ellie.shift(1, fill_value=False))
  group_id = new_group_start.cumsum()
  df['count'] = group_id.where(is_not_ellie, None)
  df.loc[~is_not_ellie, 'count'] = None
  participant_df = df[~pd.isna(df['count'])]
  group_df = participant_df.dropna(subset=['count']).groupby('count').agg(
      start_time=('start_time', 'min'), # 그룹의 가장 이른 시작 시간
      stop_time=('stop_time', 'max')   # 그룹의 가장 늦은 종료 시간
  ).reset_index()
  return participant_df, group_df

def process_vision(df):
  timestamp = df.timestamp
  au_r = df.filter(like='au').filter(like='_r')
  gz_df = df.filter(like='gz')
  gz_h = gz_df.filter(like='h')
  ps_t = df.filter(like='ps').filter(like='T')
  ps_r = df.filter(like='ps').filter(like='R')
  vision = pd.concat([timestamp, au_r, gz_h, ps_t, ps_r], axis=1)
  return vision

def read_v(zip, dataset, label_dataset):
  try:
    id, label = zip

    VISION = []

    t_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'Transcription', f'{id}_transcript.csv'))
    v_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'Vision Summary', f'{id}_vision_summary.csv'))

    _, group_df = process_transcription(t_df)
    vision = process_vision(v_df)

    for _, row in group_df.iterrows():
      start = row.start_time
      stop = row.stop_time
      v_target = vision.loc[(start <= vision.timestamp) & (vision.timestamp <= stop)]
      v_target = v_target.drop(columns=['timestamp'])
      # (seq_len, v_columns)
      v_target_list = v_target.values.tolist()

      if len(v_target_list) > 0:
        VISION.append(v_target_list)
        label_dataset.append(label)
      else:
        print(f"No Data in vision # {id}")

    # VISION: (B, seq_len, v_columns)
    dataset.extend(VISION)

  except Exception as e:
    print("오류 발생:",e)

if __name__=="__main__":
  mgr = Manager()
  train_dataset = mgr.list()
  train_label_dataset = mgr.list()
  val_dataset = mgr.list()
  val_label_dataset = mgr.list()
  
  train_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
  val_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'dev_split_Depression_AVEC2017.csv'))
  
  train_id = train_df.Participant_ID.tolist()
  val_id = val_df.Participant_ID.tolist()
  train_label = train_df.PHQ8_Binary.tolist()
  val_label = val_df.PHQ8_Binary.tolist()
  train_zip = zip(train_id, train_label)
  val_zip = zip(val_id, val_label)

  with Pool(processes=10) as p:
    with tqdm(total=len(train_id)) as pbar:
      for v in p.imap_unordered(partial(read_v, dataset=train_dataset, label_dataset=train_label_dataset), train_zip):
        pbar.update()

    with tqdm(total=len(val_id)) as pbar:
      for v in p.imap_unordered(partial(read_v, dataset=val_dataset, label_dataset=val_label_dataset),val_zip):
        pbar.update()
  print('학습 데이터 길이:',len(train_dataset))
  print('검증 데이터 길이:',len(val_dataset))
  print('input_dim:', len(train_dataset[0][0]))