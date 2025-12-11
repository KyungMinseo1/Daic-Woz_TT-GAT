## Daic-Woz-LSTM\_Graph (README.md)

### 1. 프로젝트 개요

이 프로젝트는 [DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)] 데이터셋을 활용하여 다양한 **멀티모달(Multimodal)** 및 **토픽(Topic) 기반 BiLSTM-Graph** 모델을 구현하고 실험하기 위한 저장소입니다. 주요 목표는 대화 데이터의 텍스트, 음성, 시각적 특성을 통합하고 그래프 신경망(GNN) 구조를 활용하여 모델의 성능을 향상시키는 것입니다.

-----

### 2. 환경 설정 및 의존성

프로젝트 실행을 위해 필요한 Python 라이브러리 및 환경 설정입니다.

#### 2.1. 의존성 설치

프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

**주요 라이브러리:** `PyTorch`, `torch-geometric` (Graph Neural Network), `Sentence-Transformers`, `pandas`, `numpy`

-----

### 3. 프로젝트 실행 및 데이터 준비

프로젝트의 학습을 진행하거나 데이터 구조를 확인하기 전에, 각 모델 구조에 맞는 데이터셋 파일이 올바르게 준비되었는지 확인해야 합니다. 모든 명령어는 **프로젝트의 루트 디렉토리** (`신 프로젝트/`)에서 실행해야 합니다.

#### 3.1. 데이터셋 확인 및 스크립트 실행

아래 명령어들은 `python -m` 옵션을 사용하여 파이썬 패키지 구조 내의 `dataset.py` 파일을 모듈로서 실행합니다.

| 설정 이름 | 실행 명령어 | 설명 |
| :--- | :--- | :--- |
| **Multimodal BiLSTM** | `python -m graph.multimodal_bilstm.dataset` | 일반 멀티모달 BiLSTM 설정의 데이터셋 스크립트 실행 |
| **Multimodal Proxy** | `python -m graph.multimodal_proxy.dataset` | 멀티모달 프록시(Proxy) 설정의 데이터셋 스크립트 실행 |
| **Multimodal Topic BiLSTM** | `python -m graph.multimodal_topic_bilstm.dataset` | 토픽 기반 멀티모달 BiLSTM 설정의 데이터셋 스크립트 실행 |
| **Multimodal Topic BiLSTM Proxy** | `python -m graph.multimodal_topic_bilstm_proxy.dataset` | 토픽 BiLSTM 프록시 설정의 데이터셋 스크립트 실행 |
| **Multimodal Topic Proxy** | `python -m graph.multimodal_topic_proxy.dataset` | 멀티모달 토픽 프록시 설정의 데이터셋 스크립트 실행 |

#### 3.2. 학습 스크립트 실행

각 모델 폴더 내의 `train.py` 파일을 실행하여 모델 학습을 시작합니다.

```bash
# 예시: Multimodal BiLSTM 모델 학습 실행
python -m graph.multimodal_bilstm.train
```

혹은 `optuna`를 사용하여 학습을 진행할 수도 있습니다.

```bash
# 예시: Multimodal BiLSTM 모델 Optuna 학습 실행
python optuna_train/optuna_graph.py --mode multimodal_bilstm --save_dir checkpoints_optuna --save_dir_ multimodal_bilstm
```
-----

### 4. 프로젝트 구조 (개요)

프로젝트 구조는 모듈화 및 재사용성을 고려하여 설계되었습니다.

```
신 프로젝트/
├── graph/                        # 그래프 생성/학습 핵심 코드 및 패키지
│   ├── configs/                  # 모델 하이퍼파라미터 및 데이터 설정 파일
│   ├── extra/                    # 추가 스크립트
│   ├── multimodal_bilstm/        # BiLSTM 기반 모델 구현 모듈
│   ├── multimodal_proxy/         # Proxy 기반 모델 구현 모듈
│   ├── ...
│   ├── __init__.py               # Python 패키지 초기화 파일
│   └── path_config.py            # 전역 변수 및 데이터 경로 설정 파일
│
├── optuna_train/                 # Optuna 학습
│   ├── optuna_graph.py           # Optuna 학습 스크립트
│   ├── optuna_search_grid.yaml   # Optuna 탐색 범위 지정 및 하이퍼파라미터 설정
│   └── path_config.py            # 전역 변수 및 데이터 경로 설정 파일
│
├── checkpoints_optuna/           # 학습된 모델 체크포인트 저장소
├── data/                         # 원본 데이터 및 전처리된 데이터 저장소
└── requirements.txt              # 의존성 목록
```

-----
