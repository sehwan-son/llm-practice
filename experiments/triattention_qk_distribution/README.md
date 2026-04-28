# Triattention Q/K Distribution

Inference 중 attention layer의 pre-RoPE `Q`/`K`를 캡처하고, Qwen 계열 `rotate_half` RoPE pair를 frequency band별 complex cloud로 그리는 실험입니다.

남긴 기능은 아래 여섯 가지입니다.

1. head별 pre-RoPE `Q`/`K` 수집
2. frequency band별 Q/K complex plot 생성
3. 논문 Appendix B.7의 dominant frequency score `C_f = E[|q_f|] * E[|k_f|]` 계산
4. 논문 Figure 2(C)처럼 attention head별 dominant band의 Q/K concentration `R` 분포 plot 생성
5. layer별로 모든 query head의 top-1 dominant band Q/K complex plot 생성
6. KVQuant Figure 2의 왼쪽 그림처럼 pre-RoPE Key의 token-by-channel magnitude heatmap/3D surface 생성

## Files

- [analyze_pre_rope_qk.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/analyze_pre_rope_qk.py:1): CLI entrypoint
- [run_analysis.sh](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/run_analysis.sh:1): 자주 쓰는 기본값으로 실행하는 wrapper
- [clean_outputs.sh](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/clean_outputs.sh:1): 기존 output 결과 삭제용 script
- [qk_rope_analysis/cli.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/cli.py:1): CLI argument 정의
- [qk_rope_analysis/config.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/config.py:1): device/dtype, prompt loading, AIME2025 auto download
- [qk_rope_analysis/modeling.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/modeling.py:1): model/tokenizer loading, pre-RoPE Q/K hook
- [qk_rope_analysis/complex_pairs.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/complex_pairs.py:1): split-half RoPE pair to complex tensor
- [qk_rope_analysis/dominant_bands.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/dominant_bands.py:1): dominant frequency band scoring
- [qk_rope_analysis/plotting_common.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/plotting_common.py:1): matplotlib loading, sampling, plotting utility
- [qk_rope_analysis/key_magnitude_plots.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/key_magnitude_plots.py:1): pre-RoPE Key magnitude heatmap/3D surface plot
- [qk_rope_analysis/qk_cloud_plots.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/qk_cloud_plots.py:1): Q/K frequency cloud, top band, concentration plot
- [qk_rope_analysis/plotting.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/plotting.py:1): old imports compatibility facade
- [qk_rope_analysis/workflow.py](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/qk_rope_analysis/workflow.py:1): capture, analysis, export orchestration

## RoPE Pair

Qwen의 `rotate_half` 구현에 맞춰 adjacent pair가 아니라 split-half pair를 씁니다.

```text
complex_pair[p] = x[p] + i * x[p + head_dim/2]
```

head dimension이 `128`이면 band `0`은 dim `(0, 64)`, band `1`은 `(1, 65)`입니다.

## Run

기본 calibration input은 [calibration/aime2025.jsonl](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/calibration/aime2025.jsonl)입니다. 파일이 없으면 Hugging Face의 `zai-org/glm-simple-evals-dataset`에서 AIME2025 JSONL을 내려받고 `question` 필드를 이어붙여 forward pass에 넣습니다.

```bash
bash experiments/triattention_qk_distribution/run_analysis.sh --max-length 10000
```

`run_analysis.sh`는 기본으로 `--device cuda:1,cuda:0`을 사용합니다. 즉 GPU 1을 먼저 시도하고, unavailable이면 GPU 0으로 fallback합니다. 다른 장치를 쓰려면 환경변수나 CLI flag로 덮어씁니다.

```bash
DEVICE=cuda:0 bash experiments/triattention_qk_distribution/run_analysis.sh
bash experiments/triattention_qk_distribution/run_analysis.sh --device cpu
bash experiments/triattention_qk_distribution/run_analysis.sh --device cuda:2,cuda:0
```

직접 실행할 수도 있습니다.

```bash
python experiments/triattention_qk_distribution/analyze_pre_rope_qk.py \
  --model experiments/triattention_qk_distribution/models \
  --device cuda:1,cuda:0 \
  --layers 0 \
  --heads 0 \
  --max-length 10000 \
  --output-dir experiments/triattention_qk_distribution/outputs/manual_run
```

여러 layer/head를 보려면 comma list 또는 `all`을 씁니다.

```bash
LAYERS=all HEADS=all OUTPUT_DIR=experiments/triattention_qk_distribution/outputs/full_scan \
bash experiments/triattention_qk_distribution/run_analysis.sh --max-length 10000
```

별도 calibration 파일을 쓰려면 plain text, JSON, JSONL을 넘길 수 있습니다.

```bash
python experiments/triattention_qk_distribution/analyze_pre_rope_qk.py \
  --prompt-file path/to/calibration.jsonl \
  --prompt-field problem \
  --layers 0 \
  --heads 0
```

## Outputs

출력 디렉토리에는 아래 파일만 생성합니다.

- `metadata.json`: 실행 설정, prompt source, token count, RoPE pairing 정보
- `dominant_frequency_bands.csv`: layer/query head별 top-K dominant frequency band
- `qk_concentration_by_head.csv`: layer/query head별 dominant band의 Q/K concentration `R`
- `concentration_distribution/qk_concentration_r_distribution.png`: Q/K concentration `R` histogram
- `pre_rope_key_magnitude/k_layer*_pre_rope_key_magnitude.png`: pre-RoPE Key의 token-by-channel magnitude heatmap
- `pre_rope_key_magnitude_3d/k_layer*_pre_rope_key_magnitude_3d.png`: 논문 Figure 2 스타일의 pre-RoPE Key magnitude 3D surface
- `frequency_grids/qk_layer*_qhead*_kvhead*_frequency_grid.png`: band별 Q/K complex cloud plot
- `top_frequency_bands/qk_layer*_qhead*_kvhead*_top_bands.png`: dominant top-K band만 모은 Q/K complex cloud plot
- `top1_heads_by_layer/qk_layer*_top1_heads.png`: layer별 모든 query head의 top-1 dominant band Q/K complex cloud plot
- `complex_pairs.pt`: `--save-complex-tensors` 사용 시 저장되는 complex Q/K tensor

`dominant_frequency_bands.csv`의 `score`는 `E[|q_f|] * E[|k_f|]`이고, `score_share`는 해당 head 안에서 전체 band score 대비 비율입니다. GQA 모델은 query head를 대응되는 key/value head로 자동 매핑합니다.
`qk_concentration_by_head.csv`의 `q_concentration_r`, `k_concentration_r`는 각 head의 dominant band에서 `R_f = |E[x_f]| / E[|x_f|]`로 계산합니다.
Key magnitude heatmap은 captured pre-RoPE `K`를 `[token, kv_head * head_dim]`으로 펼친 뒤 `abs(K)`를 그립니다. `--key-magnitude-max-tokens`, `--key-magnitude-max-channels`로 큰 입력을 균일 샘플링할 수 있고, `--key-magnitude-color-quantile`은 colorbar 상한을 조절합니다.
3D surface는 `--key-magnitude-plot-kind surface3d` 또는 기본값 `both`로 생성합니다. `--key-magnitude-3d-max-tokens`, `--key-magnitude-3d-max-channels`는 가독성을 위한 샘플링 크기이며, 3D channel 샘플링은 균일 channel과 평균 magnitude가 큰 outlier channel을 함께 보존합니다. `--key-magnitude-3d-elev`, `--key-magnitude-3d-azim`으로 시야각을 바꿀 수 있습니다.
각 subplot은 해당 band의 전체 Q/K 좌표 범위에 맞춰 축을 잡으므로, 큰 값이 있어도 점이 잘리지 않습니다.

## Clean

기존 결과를 지울 때는 기본 dry-run으로 먼저 확인합니다.

```bash
bash experiments/triattention_qk_distribution/clean_outputs.sh
```

실제로 삭제하려면 `--yes`를 붙입니다. [outputs/README.md](/home/sehwan/llm-practice/experiments/triattention_qk_distribution/outputs/README.md:1)는 남깁니다.

```bash
bash experiments/triattention_qk_distribution/clean_outputs.sh --yes
```
