# Triattention Q/K Distribution

Qwen-style attention module에서 pre-RoPE `Q`/`K` 텐서를 캡처하고, `rotate_half` 기준의 실제 RoPE pair `(i, i + head_dim/2)`를 복소평면 점구름으로 해석하는 실험입니다.

이 폴더는 세 가지 목적을 갖습니다.

1. 특정 layer/head의 pre-RoPE `Q`/`K`를 캡처한다.
2. pair별 에너지 분포와 복소 점구름 통계를 계산한다.
3. heatmap, layer trend, complex pair plot으로 결과를 해석한다.

## 폴더 구조

```text
triattention_qk_distribution/
├── README.md
├── analyze_pre_rope_qk.py
├── plot_pre_rope_summary.py
├── run_analysis.sh
├── qk_rope_analysis/
│   ├── __init__.py
│   ├── analysis.py
│   ├── plotting.py
│   └── workflow.py
├── outputs/
│   └── README.md
├── references/
│   └── 2604.04921v1.pdf
├── models/
│   └── Qwen3.5-27B-Q8_0.gguf
└── .venv/
```

## 파일 역할

- [analyze_pre_rope_qk.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/analyze_pre_rope_qk.py:1): 메인 CLI 진입점
- [plot_pre_rope_summary.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/plot_pre_rope_summary.py:1): 이미 저장된 `summary.json`만 다시 시각화
- [run_analysis.sh](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/run_analysis.sh:1): 자주 쓰는 기본 옵션으로 실행하는 편의 스크립트
- [qk_rope_analysis/analysis.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/analysis.py:1): 텐서 캡처, 통계 계산, CSV/JSON 직렬화
- [qk_rope_analysis/plotting.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/plotting.py:1): heatmap, trend plot, complex pair plot 생성
- [qk_rope_analysis/workflow.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/workflow.py:1): 분석 파이프라인 orchestration

## 핵심 아이디어

이 실험은 adjacent pair `(2i, 2i+1)`가 아니라 Qwen3 `rotate_half` 구현에 맞는 split-half pair `(i, i + head_dim/2)`를 사용합니다.

즉 head_dim이 `128`이면:

- pair `0`은 `(0, 64)`
- pair `1`은 `(1, 65)`
- ...

복소수 표현은 아래처럼 정의합니다.

```text
complex_pair[p] = x[p] + i * x[p + head_dim/2]
```

## 실행 방법

### 1. 메인 CLI로 직접 실행

```bash
python experiments/triattention_qk_distribution/analyze_pre_rope_qk.py \
  --model Qwen/Qwen3-1.7B \
  --tensor both \
  --layers 0 \
  --heads 0 \
  --plot-summary \
  --export-csv
```

### 2. 편의 스크립트로 실행

```bash
bash experiments/triattention_qk_distribution/run_analysis.sh
```

환경변수로 기본값을 덮어쓸 수 있습니다.

```bash
MODEL=Qwen/Qwen3-1.7B \
LAYERS=all \
HEADS=all \
OUTPUT_DIR=experiments/triattention_qk_distribution/outputs/full_scan \
bash experiments/triattention_qk_distribution/run_analysis.sh --plot
```

### 3. 저장된 summary만 다시 플롯

```bash
python experiments/triattention_qk_distribution/plot_pre_rope_summary.py \
  --summary-json experiments/triattention_qk_distribution/outputs/runs/Qwen__Qwen3-1.7B/summary.json \
  --export-csv
```

## 주요 옵션

- `--model`: Hugging Face model id 또는 로컬 모델 경로
- `--tensor {q,k,both}`: 분석 대상 텐서
- `--layers 0,1,2` 또는 `--layers all`: 캡처할 레이어
- `--heads 0,5,10` 또는 `--heads all`: 분석할 헤드
- `--plot`: head별 complex pair grid와 centroid plot 생성
- `--plot-summary`: layer/head heatmap과 layer trend plot 생성
- `--export-csv`: `head_metrics.csv`, `layer_metrics.csv` 생성
- `--save-complex-tensors`: `complex_pairs.pt` 저장

## 결과 위치

기본 출력 경로는 아래 둘 중 하나입니다.

- 메인 CLI 기본값: `experiments/triattention_qk_distribution/outputs/runs/<sanitized-model-name>/`
- 편의 스크립트 기본값: `experiments/triattention_qk_distribution/outputs/manual_run/`

생성되는 주요 파일은 다음과 같습니다.

- `metadata.json`: 실행 설정, prompt, token 정보
- `summary.json`: 레이어/헤드별 요약 통계
- `head_metrics.csv`: head 단위 평탄화 지표
- `layer_metrics.csv`: layer 단위 집계 지표
- `summary_plots/*.png`: heatmap, layer trend
- `q_layer*_head*_pair_grid.png`, `k_layer*_head*_pair_grid.png`: pair별 complex cloud
- `*_centroids.png`: pair centroid 위치
- `complex_pairs.pt`: 저장 옵션 사용 시 complex tensor dump

## 결과 해석 방법

### 1. `top1_band_share`

가장 강한 RoPE pair 하나가 전체 에너지에서 차지하는 비율입니다.

- 높을수록: 특정 주파수 band 하나에 에너지가 몰려 있음
- 낮을수록: 여러 pair에 에너지가 퍼져 있음

### 2. `top4_band_share`

상위 4개 pair의 누적 점유율입니다.

- 높을수록: 몇 개의 band만으로 head 특성이 설명됨
- 낮을수록: 분포가 더 넓고 균등함

### 3. `band_entropy`

pair 에너지 분포의 정규화 entropy입니다.

- 낮을수록: 소수 pair에 집중
- 높을수록: 다수 pair에 분산

### 4. `vector_l2_mean`

token별 pre-RoPE head vector 크기 평균입니다.

- head 활성 세기의 대략적인 기준선으로 볼 수 있음
- layer별로 magnitude가 어디서 커지는지 비교할 때 유용

### 5. `center_radius`

복소 점구름 평균점이 원점에서 얼마나 떨어져 있는지 보여줍니다.

- 크면: 복소평면에서 bias된 중심이 존재
- 작으면: 원점 주변에 더 균형적으로 분포

### 6. `axis_ratio`

복소 점구름 공분산의 장축/단축 비율입니다.

- `1`에 가까우면: 원형에 가까운 cloud
- 크면: 한 축으로 길게 늘어난 anisotropic cloud

### 7. `summary_plots/*_head_pair_energy.png`

한 레이어 안에서 각 head가 어떤 pair에 에너지를 쓰는지 heatmap으로 보여줍니다.

- 세로 비교: head 간 band 선호 차이
- 가로 비교: 어떤 pair가 지속적으로 지배적인지

### 8. `*_pair_grid.png`

선택한 head의 pair별 복소 점구름을 직접 보여줍니다.

- cloud가 조밀하면 해당 pair 값 범위가 안정적
- 퍼져 있으면 분산이 큼
- centroid가 많이 이동하면 평균 위상이 쏠려 있음

## 추천 워크플로

1. `--layers all --heads all --plot-summary --export-csv`로 전체 경향을 본다.
2. heatmap에서 이상한 layer/head를 찾는다.
3. 그 head만 `--plot`으로 다시 실행해 complex pair grid를 확인한다.
4. 필요하면 `summary.json`을 저장해 두고 `plot_pre_rope_summary.py`로 재시각화한다.

## 참고

- 현재 폴더는 생성 결과물을 비운 깨끗한 상태로 정리되어 있습니다.
- 새 실행 결과는 `outputs/` 아래에만 생성되도록 맞춰져 있습니다.
- 이 폴더 안 `.venv/`는 로컬 실행 환경이며 구조 설명에는 포함했지만 분석 코드 자체와는 분리된 요소입니다.
