# Performance Estimation Accuracy Evaluation

Evaluates the accuracy of ShuntServe's profiling-free roofline model estimator by comparing predicted vs measured throughput across different batch sizes, TP/PP configurations, models, and GPU types.

## Experiments

### Experiment 1: TP/PP Variation
Same physical 4 GPUs with different TP/PP combinations to isolate parallelism effects:

| Config | Instance(s) | TP | PP |
|--------|------------|----|----|
| TP4-PP1 | g6e.12xlarge (×1 stage) | 4 | 1 |
| TP2-PP2 | g6e.12xlarge(half) (×2 stages) | 2 | 2 |
| TP1-PP4 | g6e.xlarge (×4 stages) | 1 | 4 |

### Experiment 2: Different Model / GPU
| Config | Instance | GPU | Model | TP | Layers |
|--------|----------|-----|-------|----|--------|
| A | g6e.12xlarge | L40S×4 | Llama-3.1-70B | 4 | 80 |
| B | g6.12xlarge | L4×4 | Qwen2.5-32B | 4 | 64 |

For each config:
- **Predicted**: `predict.py` generates estimator predictions
- **Measured**: `measure.py` runs actual vLLM inference throughput measurements

## File Structure

```
PerformanceEstimation/
├── README.md           # This file
├── estimator/
│   ├── predict.py      # Estimator prediction (runs locally, no GPU needed)
│   ├── measure.py      # Actual throughput measurement (requires AWS instances)
│   ├── plot.py         # Result comparison figure generation
│   ├── nodes.py        # Instance IPs (fill before measurement)
│   └── results/        # Output directory (predictions and measurements)
```

## Usage

### 1. Generate Predictions (local)
```bash
# TP/PP variations: 70B on L40S×4 (TP=4/2/1)
python estimator/predict.py --model meta-llama/Llama-3.1-70B-Instruct --tp-variations g6e

# TP/PP variations: 8B on L4×4 (TP=4/2/1)
python estimator/predict.py --model meta-llama/Llama-3.1-8B-Instruct --tp-variations g6

# Single instance
python estimator/predict.py --model meta-llama/Llama-3.1-70B-Instruct --instance g6e.12xlarge
python estimator/predict.py --model meta-llama/Llama-3.1-8B-Instruct --instance g6.xlarge
```

### 2. Run Measurements (AWS)
```bash
# Fill IPs in estimator/nodes.py first
python estimator/measure.py --config 70B_L40S
python estimator/measure.py --config 8B_L4
```

### 3. Generate Figures
```bash
python estimator/plot.py
```
