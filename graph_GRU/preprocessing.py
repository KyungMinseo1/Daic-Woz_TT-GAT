import sys
import pandas as pd
import numpy as np
from loguru import logger

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

BACKCHANNELS = {
  "okay_confirm (okay)", "yeah_downer (yeah)", "wild_laughter5 ((laughter))", "mhm (mhm)", 
  "yeah3 (yeah)", "thats_great (that's great)", "nice (nice)", "uh_huh (uh huh)", 
  "thats_good (that's good)", "mm (mm)", "really (really)", "awesome (awesome)", "im_sorry (i'm sorry)",
  "wow (wow)", "oh_no (oh no)", "Ellie17Dec2012_02 (uh huh)", "see_mean (i see what you mean)", 
  "yeah2 (yeah)", "yeah1 (yeah)", "yes (yes)", "hmm1 (hmm)", "right2 (right)", "right (right)", 
  "hmmB (hmm)", "isee_downer (i see)", "um (um)", "aw (aw)", "okay", "yeah", "[laughter]", 
  "mhm", "that's great", "nice", "uh huh", "that's good", "mm", "really", "awesome", 
  "wow", "oh no", "right", "hmm", "i see", "um", "aw", "cool", "oh my gosh,"
}

def process_transcription(df, time_interval):
  utterances = []
  topics = []
  start_stop_list = []
  start_stop_list_ellie = []

  temp_text = ""
  start_time = None
  stop_time = None
  current_topic = None

  for idx, row in df.iterrows():
    if pd.isna(row['topic']):
      continue
    speaker = row['speaker']
    value = str(row['value']).strip()

    if speaker == 'Ellie':
      is_backchannel = value.lower() in [b.lower() for b in BACKCHANNELS]

      start_stop_list_ellie.append([row['start_time'], row['stop_time']])

      if not is_backchannel:
        # 호응 발화가 아닐 경우 -> 이전 데이터 저장 후 초기화
        if temp_text != "":
          utterances.append(temp_text.strip())
          topics.append(current_topic)
          start_stop_list.append([start_time, stop_time])
          temp_text = ""
          start_time = None
      else:
        # 호응 발화일 경우 -> 기존 데이터에 이어서 저장
        continue

    elif speaker == 'Participant':
      if start_time is None:
        start_time = row['start_time']
        current_topic = row['topic']

      if (row['stop_time'] - start_time) > time_interval:
        # 연속 발화 길이가 'time_interval'초를 넘는 경우 -> 데이터 강제 저장 (노드 분리)
        utterances.append((temp_text + " " + value).strip())
        topics.append(row['topic'])
        start_stop_list.append([start_time, row['stop_time']])

        temp_text = ""
        start_time = None

      elif current_topic != row['topic'] and temp_text != "":
        # 연속 발화 주제가 다른 경우 -> 데이터 강제 저장 (노드 분리)
        utterances.append(temp_text.strip())
        topics.append(current_topic)
        start_stop_list.append([start_time, stop_time])

        temp_text = value + ". "
        start_time = row['start_time']
        stop_time = row['stop_time']
        current_topic = row['topic']

      else:
        # 어떤 경우에도 해당되지 않을 경우 -> 데이터 연속 저장
        temp_text += value + ". "
        stop_time = row['stop_time']
        current_topic = row['topic']

  # 마지막 값 저장
  if temp_text != "":
    utterances.append(temp_text.strip())
    topics.append(current_topic)
    start_stop_list.append([start_time, stop_time])

  return utterances, topics, start_stop_list, start_stop_list_ellie

def process_vision(df, start_stop_list_ellie):
  keep_mask = np.ones(len(df), dtype=bool)
  timestamp_values = df['timestamp'].values
  for start, stop in start_stop_list_ellie:
    condition = (start <= timestamp_values) & (timestamp_values <= stop)
    keep_mask[condition] = False
  timestamp = df.timestamp
  # ft_x = df.filter(like='ftx')
  # ft_y = df.filter(like='fty')
  au_r = df.filter(like='auAU').filter(like='_r')
  gz_df = df.filter(like='gz')
  # gz_h = gz_df.filter(like='h')
  ps_t = df.filter(like='ps').filter(like='T')
  ps_r = df.filter(like='ps').filter(like='R')
  # features = pd.concat([au_r, gz_df, ft_x, ft_y, ps_t, ps_r], axis=1)
  features = pd.concat([au_r, gz_df, ps_t, ps_r], axis=1)
  if features.shape[1] == 0:
    logger.warning("No vision features found! Adding a dummy feature.")
    features['dummy_feature'] = 0.0
  vision = pd.concat([timestamp, features], axis=1)
  return vision[keep_mask]

def process_audio(df, start_stop_list_ellie):
  df = df.reset_index()
  keep_mask = np.ones(len(df), dtype=bool)
  idx_values = df['index'].values
  for start, stop in start_stop_list_ellie:
    condition = (start * 100 <= idx_values) & (idx_values <= stop * 100)
    keep_mask[condition] = False
  renamed_df = df.rename(
      columns={
        '0':'F0', '1': 'VUV',
        '2':'NAQ', '3':'QOQ',
        '4':'H1H2', '5':'PSP',
        '6':'MDQ', '7':'peakSlope',
        '8':'Rd', '74':'F1',
        '75':'F2', '76': 'F3', '77': 'F4', '78': 'F5'
      }
    )
  timestamp = renamed_df[['index']]
  audio1 = renamed_df[['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd']]
  audio2 = renamed_df[['F1', 'F2', 'F3', 'F4', 'F5']]
  audio = pd.concat([audio1, audio2], axis=1)
  audio.loc[audio.VUV==0] = 0
  final_audio = pd.concat([timestamp, audio], axis=1)
  return final_audio[keep_mask]