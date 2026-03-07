# Performance Estimation Accuracy Evaluation

## 목적
ShuntServe의 profiling-free roofline model estimator가 다양한 batch size에서
throughput scaling trend를 정확히 포착하는지 검증한다.

## 실험 구성

### Experiment 1: TP/PP Variation (교수님 코멘트 대응)
같은 물리 GPU 4장(g6e.12xlarge)에서 TP/PP 조합을 바꿔가며 예측 vs 실측 비교:

| Config | Instance(s) | TP | PP | 설명 |
|--------|------------|----|----|------|
| TP4-PP1 | g6e.12xlarge (×1 stage) | 4 | 1 | 전체 TP |
| TP2-PP2 | g6e.12xlarge(half) (×2 stages) | 2 | 2 | 혼합 |
| TP1-PP4 | g6e.xlarge (×4 stages) | 1 | 4 | 전체 PP |

### Experiment 2: Different Model/GPU
| Config | Instance | GPU | Model | TP | Layers |
|--------|----------|-----|-------|----|--------|
| A | g6e.12xlarge | L40S×4 | Llama-3.1-70B | 4 | 80 |
| B | g6.12xlarge | L4×4 | Qwen2.5-32B | 4 | 64 |

각 config에서:
- **Predicted**: `predict.py`로 estimator 예측값 산출
- **Measured**: `measure.py`로 실제 vLLM inference throughput 측정

## 파일 구조

```
PerformanceEstimation/
├── README.md           # 이 파일
├── nodes.py            # 실험 인스턴스 IP (실험 전 채울 것)
├── predict.py          # Estimator 예측값 산출 (로컬 실행 가능)
├── measure.py          # 실제 throughput 측정 (AWS 인스턴스 필요)
├── plot.py             # 결과 비교 figure 생성
└── results/            # 결과 저장 디렉토리 (자동 생성)
```

## 실행 순서

### 1. Predicted 값 산출 (로컬)
```bash
# TP/PP variations: 70B on L40S×4 (TP=4/2/1)
python predict.py --model meta-llama/Llama-3.1-70B-Instruct --tp-variations g6e

# TP/PP variations: 8B on L4×4 (TP=4/2/1)
python predict.py --model meta-llama/Llama-3.1-8B-Instruct --tp-variations g6

# Single instance
python predict.py --model meta-llama/Llama-3.1-70B-Instruct --instance g6e.12xlarge
python predict.py --model meta-llama/Llama-3.1-8B-Instruct --instance g6.xlarge
```

### 2. Measured 값 측정 (AWS)
```bash
# nodes.py에 IP 채운 후
python measure.py --config 70B_L40S
python measure.py --config 8B_L4
```

### 3. Figure 생성
```bash
python plot.py
```
