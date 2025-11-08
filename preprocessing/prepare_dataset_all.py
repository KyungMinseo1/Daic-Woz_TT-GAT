import os, config
import pandas as pd
import numpy as np
import torch
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

def read_tva(id, dataset):
  try:
    TRANSCRIPTION = []
    VISION = []
    AUDIO = []
    t_df = pd.read_csv(os.path.join(config.DATA_DIR, 'Transcription', f'{id}_transcript.csv'))
    v_df = pd.read_csv(os.path.join(config.DATA_DIR, 'Vision Summary', f'{id}_vision_summary.csv'))
    a_df = pd.read_csv(os.path.join(config.DATA_DIR, 'Audio Summary', f'{id}_audio_summary.csv'))

    participant_df, group_df = process_transcription(t_df)
    vision = process_vision(v_df)

    for _, row in group_df.iterrows():
      start = row.start_time
      stop = row.stop_time
      t_target = participant_df.loc[participant_df['count']==row['count']]
      v_target = vision.loc[(start <= vision.timestamp) & (vision.timestamp <= stop)]
      v_target = v_target.drop(columns=['timestamp'])
      v_target_list = v_target.values.tolist()
      a_target = a_df.iloc[int(start*100):int((stop*100)+1)]
      a_target_list = a_target.values.tolist()
      if len(t_target) > 0:
        TRANSCRIPTION.append(t_target)
      else:
        print("ERROR in transcription")
      if len(v_target_list) > 0:
        VISION.append(v_target_list)
      else:
        print("ERROR in vision")
      if len(a_target_list) > 0:
        AUDIO.append(a_target_list)
      else:
        print("ERROR in vision")
    if len(TRANSCRIPTION) == len(VISION) == len(AUDIO):
      dataset.append((TRANSCRIPTION, VISION, AUDIO))
    else:
      print("Length Not Matched")
  except Exception as e:
    print("오류 발생:",e)

if __name__=="__main__":
  mgr = Manager()
  dataset = mgr.list()
  raw_id = os.listdir(config.RAW_DATA_DIR)
  id = [i.split('_')[0] for i in raw_id]
  with Pool(processes=10) as p:
    with tqdm(total=len(id)) as pbar:
      for v in p.imap_unordered(partial(read_tva, dataset=dataset),id):
        pbar.update()
  print(len(dataset))