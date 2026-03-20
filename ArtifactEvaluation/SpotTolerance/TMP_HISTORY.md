# SpotTolerance Benchmark Migration Progress

Legacy 벤치마크 스크립트(하드코딩)를 JSON config 기반으로 마이그레이션하는 작업 기록.

## Architecture

### JSON Config Files (공통, SpotTolerance/ 루트)
- `spot_trace_events_scenario_A.json` / `_scenario_B.json` — spot interruption/restore 이벤트 정의
- `nodes_scenario_A.json` / `nodes_scenario_B.json` — 노드 이름 → IP 매핑 (실행 전 채워야 함)

### Per-Model Config (각 모델 디렉토리)
- `pipelines_{model}_scenario_A.json` / `_scenario_B.json` — 파이프라인 config + node_layer_mapping

### Directory Structure
```
SpotTolerance/
├── spot_trace_events_scenario_{A,B}.json
├── nodes_scenario_{A,B}.json
├── llama3-70b/
│   ├── pipelines_llama3_70b_scenario_{A,B}.json
│   ├── offline/
│   │   ├── scenario_A/
│   │   │   ├── shuntserve.py   ✅
│   │   │   └── show_events.py  ✅
│   │   └── scenario_B/
│   │       ├── shuntserve.py   ✅
│   │       └── show_events.py  ✅
│   └── online/
│       ├── scenario_A/
│       └── scenario_B/
├── qwen3-32b/
│   ├── pipelines_qwen3_32b_scenario_{A,B}.json
│   ├── offline/
│   │   ├── scenario_A/
│   │   └── scenario_B/
│   └── online/
│       ├── scenario_A/
│       └── scenario_B/
└── legacy/                    (이전 하드코딩 버전, 참고용)
```

## Design Decisions

1. **Pipeline config**: pipelines JSON의 `config` dict를 그대로 사용 (ModelPlacement 처럼 `stages`에서 빌드하지 않음)
2. **Node IP resolution**: `nodes_scenario_X.json`에서 노드 이름 → IP 조회
3. **Spot ↔ On-demand mapping**: `spot_` ↔ `on_demand_` prefix 치환 (`get_counterpart_name()`)
4. **Same-time event grouping**: 같은 `time_min`의 이벤트들을 `defaultdict(list)`로 그룹핑하여 단일 `switch_nodes` 호출로 합침
5. **Initial on-demand pre-population**: 초기 파이프라인에 `on_demand_` 노드가 있으면 해당 spot counterpart를 `interrupted_spots`에 미리 등록 (restore 이벤트 처리를 위해 필수)
6. **Results 저장**: `save_benchmark_results()` 사용 (`ArtifactEvaluation/ModelPlacement/save_results.py`)
7. **Reference 코드 스타일**: `ModelPlacement/offline/llama3-70b/shuntserve.py` 최신 패턴 준수

## Strategy Scripts (per scenario)

| Script | GlobalServer mode | Event handling | 설명 |
|--------|------------------|----------------|------|
| `shuntserve.py` | `migration` | `switch_nodes()` | Hot migration (ShuntServe 방식) |
| `concurrent_initialization.py` | `re-routing` | `switch_nodes()` | Re-routing + hot switch |
| `request_migration.py` | `migration` | `stop_nodes()` + `create_pipeline()` | Cold restart with migration |
| `no_handle.py` | `re-routing` | `stop_nodes()` + `create_pipeline()` | Cold restart, no migration |
| `only_ondemand.py` | default | 없음 | On-demand only baseline (이벤트 없이 실행) |
| `warmup.py` | default | 없음 | 모든 노드 pre-provision (이벤트 없이 실행) |

## GlobalServer 변경사항

### `max_duration` 파라미터 추가
- `GlobalServer/evaluation_utils.py` — `run_trace_replay_benchmark()`에 `max_duration: float = None` 추가
  - `asyncio.wait(tasks, timeout=max_duration)`으로 시간 초과 시 pending 태스크 cancel, 완료된 결과만 수집
- `GlobalServer/benchmark_utils.py` — `run_trace_benchmark()`에 `max_duration` 파라미터 추가 및 전달
- 기존 호출자는 `max_duration=None` (default)으로 동작 변화 없음

## Completed

- [x] `llama3-70b/offline/scenario_A/shuntserve.py`
- [x] `llama3-70b/offline/scenario_B/shuntserve.py`
- [x] `llama3-70b/offline/scenario_A/show_events.py` — 이벤트 매핑 디버깅 도구
- [x] `llama3-70b/offline/scenario_B/show_events.py` — 이벤트 매핑 디버깅 도구
- [x] `max_duration` 지원 (`evaluation_utils.py`, `benchmark_utils.py`)

## TODO

### llama3-70b/offline
- [ ] `scenario_{A,B}/no_handle.py`
- [ ] `scenario_{A,B}/request_migration.py`
- [ ] `scenario_{A,B}/concurrent_initialization.py`
- [ ] `scenario_{A,B}/only_ondemand.py`
- [ ] `scenario_{A,B}/warmup.py`

### llama3-70b/online
- [ ] scenario_A, scenario_B 전체 (offline과 동일 구조, `time_scale` 변경)

### qwen3-32b/offline
- [ ] scenario_A, scenario_B 전체

### qwen3-32b/online
- [ ] scenario_A, scenario_B 전체
