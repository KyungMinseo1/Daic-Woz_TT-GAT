# Daic-Woz-LSTM-Graph

DAIC-WOZ 데이터셋을 활용하여 BiLSTM/GRU 및 그래프 신경망(GNN) 기반의 멀티모달 우울증 탐지를 구현한 공식 저장소입니다.

본 프로젝트는 [DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)] 데이터셋의 텍스트, 오디오, 비디오 특징을 통합합니다. 임상 인터뷰 내의 시간적 및 의미론적 관계를 모델링하기 위해 토픽 기반 그래프 구조를 사용합니다.

---

## 🚀 파이프라인 개요 (Pipeline Overview)

프로젝트는 데이터 준비부터 심층 분석까지 순차적인 파이프라인을 따릅니다:

1.  **데이터셋 획득**: 원본 DAIC-WOZ 데이터셋을 다운로드합니다.
2.  **전처리**: `notebooks/data_process.ipynb`를 사용하여 데이터를 정제하고 포맷팅합니다.
3.  **토픽 분류**: `notebooks/topic.ipynb`를 통해 LLM을 활용하여 대화 토픽을 분류합니다.
4.  **모델 학습**: BiLSTM(`graph/`) 또는 GRU(`graph_GRU/`) 모듈을 사용하여 실험을 수행합니다.
5.  **하이퍼파라미터 최적화**: `optuna_train/`에서 [Optuna](https://optuna.org/)를 사용하여 실험을 진행합니다.
6.  **심층 분석**: `graph_explanation/`을 사용하여 모델 성능 및 설명 가능성을 평가합니다.

---

## 🛠️ 요구 사항 및 설정 (Requirements & Setup)

### 환경 (Environment)
- **언어**: Python 3.10+
- **프레임워크**: PyTorch, PyTorch Geometric (PyG), Optuna, Sentence-Transformers
- **패키지 매니저**: pip

### 설치 (Installation)
```bash
pip install -r requirements.txt
```
*참고: PyTorch 및 PyG 호환성을 위해 적절한 CUDA 버전이 설치되어 있는지 확인하십시오.*

### 설정 (.env)
루트 디렉토리에 `.env` 파일을 생성하고 토픽 분류를 위한 OpenAI API 키를 추가합니다:
```text
OPENAI_API_KEY=your_api_key_here
```

---

## 📊 데이터 준비 (Data Preparation)

1.  **원본 데이터셋**: DAIC-WOZ 데이터셋이 `data/` 디렉토리에 있는지 확인합니다.
2.  **전처리**: `notebooks/data_process.ipynb`를 실행하여 원본 전사본(transcripts)과 멀티모달 특징을 처리합니다.
3.  **토픽 라벨링**: `notebooks/topic.ipynb`를 실행하여 LLM 기반 토픽 추출을 수행합니다. 이 단계는 토픽 기반 그래프 구축에 필수적입니다.

---

## 🏋️ 학습 및 실험 (Training & Experiments)

개별 모듈을 학습시키거나 하이퍼파라미터 최적화를 실행할 수 있습니다.

### 단일 모델 학습
BiLSTM 또는 GRU 모듈에 대한 학습 스크립트를 실행합니다.

#### 예시: Multimodal Topic BiLSTM Proxy
```bash
python -m graph.multimodal_topic_bilstm_proxy.train --num_epochs 100 --config graph/configs/architecture_TT_GAT.yaml --save_dir checkpoints --save_dir_ topic_bilstm_proxy
```

#### 예시: Multimodal Topic GRU Proxy
```bash
python -m graph_GRU.multimodal_topic_gru_proxy.train --num_epochs 100 --config graph_GRU/configs/architecture_TT_GAT.yaml --save_dir checkpoints --save_dir_ topic_gru_proxy
```

### 인자 사용법 (Parse Args)
프록시 모듈에서 흔히 사용되는 인자들:
- `--num_epochs`: 학습 에포크 수 (기본값: 100).
- `--config`: YAML 설정 파일 경로.
- `--resume`: 학습을 재개할 체크포인트 경로.
- `--save_dir`: 체크포인트 저장을 위한 기본 디렉토리.
- `--save_dir_`: 현재 실행을 위한 세부 하위 디렉토리.

### Optuna 최적화
자동 하이퍼파라미터 탐색을 수행하려면:
- **BiLSTM**: `python optuna_train/optuna_graph.py`
- **GRU**: `python optuna_train/optuna_graph_gru.py`

---

## ⚙️ 설정 (Configurations)

모델 아키텍처 및 탐색 공간은 YAML 파일을 통해 관리됩니다:

| 유형 | 설정 파일 | 설명 |
| :--- | :--- | :--- |
| **BiLSTM 아키텍처** | `graph/configs/architecture_TT_GAT.yaml` | LSTM-GNN 모델을 위한 표준 아키텍처. |
| **GRU 아키텍처** | `graph_GRU/configs/architecture_TT_GAT.yaml` | GRU-GNN 모델을 위한 표준 아키텍처. |
| **Optuna (BiLSTM)** | `optuna_train/optuna_search_grid.yaml` | BiLSTM 최적화를 위한 탐색 공간. |
| **Optuna (GRU)** | `optuna_train/optuna_search_grid_gru.yaml` | GRU 최적화를 위한 탐색 공간. |

---

## 🔍 분석 및 설명 가능성 (Analysis & Explainability, `graph_explanation/`)

모델의 심층 분석을 위해:

- **F1 점수 비교**: `graph_explanation/f1_visualization.py` (또는 `.ipynb`)를 사용하여 다양한 Optuna 학습 모델의 F1 점수를 비교합니다.
  ```bash
  python graph_explanation/f1_visualization.py --model_dir checkpoints_optuna
  ```
- **GNN Explainer**: `graph_explanation/visualization_audio_video_text.ipynb`를 사용하여 GNNExplainer를 통한 심층 분석을 수행하고, 그래프 내 오디오, 비디오, 텍스트 특징의 중요도를 시각화합니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
.
├── graph/                # BiLSTM 기반 GNN 모델
│   └── configs/          # BiLSTM을 위한 YAML 설정
├── graph_GRU/            # GRU 기반 GNN 모델
│   └── configs/          # GRU를 위한 YAML 설정
├── graph_explanation/    # 시각화 및 설명 가능성 도구
├── notebooks/            # 데이터 처리 및 토픽 분류 (Jupyter)
├── optuna_train/         # Optuna 하이퍼파라미터 최적화 스크립트
├── data/                 # 데이터셋 저장소 (DAIC-WOZ)
├── checkpoints/          # 모델 체크포인트
└── requirements.txt      # 의존성 목록
```

