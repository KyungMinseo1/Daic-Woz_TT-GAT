import torch
import sys
import os
import yaml
import pandas as pd
from loguru import logger
from collections import Counter

# 기존 프로젝트 구조에 따른 임포트
from .. import path_config
from ..graph_construct import GraphConfig
from .dataset import TopicProxyGRU_GC

# 로그 설정
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

def main():
    # 1. 설정 파일 로드 (가장 최근에 만든 best_config.yaml 사용)
    # 경로는 상황에 맞게 수정하세요 (기본값: graph_GRU/configs/best_config.yaml)
    config_rel_path = 'graph_GRU/configs/best_config.yaml'
    config_path = os.path.join(path_config.ROOT_DIR, config_rel_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return

    with open(config_path, 'r', encoding="utf-8") as ymlfile:
        config = yaml.safe_load(ymlfile)
    logger.info(f"Loaded config from {config_path}")

    # 2. 데이터 프레임 로드 (기존 train.py 로직 동일)
    logger.info("Loading CSV split files...")
    train_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
    val_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'dev_split_Depression_AVEC2017.csv'))
    test_df = pd.read_csv(os.path.join(path_config.DATA_DIR, 'full_test_split.csv'))

    # 전처리는 전체 데이터를 대상으로 수행
    train_id = train_df.Participant_ID.tolist()
    val_id = val_df.Participant_ID.tolist()
    train_label = train_df.PHQ8_Binary.tolist()
    val_label = val_df.PHQ8_Binary.tolist()

    test_id = test_df.Participant_ID.tolist()
    test_label = test_df.PHQ_Binary.tolist()

    # 3. 그래프 생성 설정
    graph_config = GraphConfig(
        model_name=config['training']['embed_model'],
        time_interval=config['training']['time_interval'],
        use_summary_node=config['model']['use_summary_node'],
        colab_path=None # RunPod 환경이므로 None
    )

    gc = TopicProxyGRU_GC()

    # 4. Train/Val 그래프 생성 (가장 오래 걸리는 작업)
    logger.info("Processing Train + Val Data (this may take several minutes)...")
    train_graphs, (t_dim, v_dim, a_dim) = gc.make_graph(
        ids = train_id + val_id,
        labels = train_label + val_label,
        config = graph_config
    )

    logger.info("Processing Test(Validation) Data...")
    val_graphs, _ = gc.make_graph(
        ids = test_id,
        labels = test_label,
        config = graph_config
    )

    # 5. 결과 저장
    # 저장 경로: /workspace/Daic-Woz_TT-GAT/preprocessed_data
    save_dir = os.path.join(path_config.ROOT_DIR, "preprocessed_data")
    os.makedirs(save_dir, exist_ok=True)

    train_save_path = os.path.join(save_dir, 'train_graphs.pt')
    val_save_path = os.path.join(save_dir, 'val_graphs.pt')

    logger.info(f"Saving preprocessed data to {save_dir}...")
    
    # 그래프 객체와 차원 정보를 함께 저장
    torch.save({
        'graphs': train_graphs,
        't_dim': t_dim,
        'v_dim': v_dim,
        'a_dim': a_dim
    }, train_save_path)

    torch.save({
        'graphs': val_graphs
    }, val_save_path)

    logger.info("✅ Preprocessing Complete!")
    logger.info(f"Train graphs saved: {train_save_path}")
    logger.info(f"Val graphs saved: {val_save_path}")
    logger.info(f"Dimensions: Text({t_dim}), Vision({v_dim}), Audio({a_dim})")

if __name__ == "__main__":
    main()