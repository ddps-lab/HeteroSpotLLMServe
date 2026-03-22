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
├── UnitTest8B/                (8B 모델 단위 테스트 — 2x g6.xlarge)
│   ├── pipelines_8b.json      ✅
│   ├── nodes.json             ✅
│   ├── spot_trace_events.json ✅
│   ├── shuntserve.py          ✅
│   ├── concurrent_initialization.py ✅
│   ├── request_migration.py   ✅
│   ├── no_handle.py           ✅
│   ├── only_ondemand.py       ✅
│   └── show_events.py         ✅
├── llama3-70b/
│   ├── pipelines_llama3_70b_scenario_{A,B}.json
│   ├── offline/
│   │   ├── scenario_A/
│   │   │   ├── shuntserve.py   ✅
│   │   │   ├── concurrent_initialization.py ✅
│   │   │   ├── request_migration.py   ✅
│   │   │   ├── no_handle.py           ✅
│   │   │   ├── only_ondemand.py       ✅
│   │   │   ├── warmup.py             ✅
│   │   │   └── show_events.py  ✅
│   │   └── scenario_B/
│   │       ├── shuntserve.py   ✅
│   │       ├── concurrent_initialization.py ✅
│   │       ├── request_migration.py   ✅
│   │       ├── no_handle.py           ✅
│   │       ├── only_ondemand.py       ✅
│   │       ├── warmup.py             ✅
│   │       └── show_events.py  ✅
│   └── online/
│       ├── scenario_A/
│       │   ├── shuntserve.py   ✅
│       │   ├── concurrent_initialization.py ✅
│       │   ├── request_migration.py   ✅
│       │   ├── no_handle.py           ✅
│       │   ├── only_ondemand.py       ✅
│       │   ├── warmup.py             ✅
│       │   └── show_events.py  ✅
│       └── scenario_B/
│           ├── shuntserve.py   ✅
│           ├── concurrent_initialization.py ✅
│           ├── request_migration.py   ✅
│           ├── no_handle.py           ✅
│           ├── only_ondemand.py       ✅
│           ├── warmup.py             ✅
│           └── show_events.py  ✅
├── qwen3-32b/
│   ├── pipelines_qwen3_32b_scenario_{A,B}.json
│   ├── offline/
│   │   ├── scenario_A/
│   │   │   ├── shuntserve.py   ✅
│   │   │   ├── concurrent_initialization.py ✅
│   │   │   ├── request_migration.py   ✅
│   │   │   ├── no_handle.py           ✅
│   │   │   ├── only_ondemand.py       ✅
│   │   │   ├── warmup.py             ✅
│   │   │   └── show_events.py  ✅
│   │   └── scenario_B/
│   │       ├── shuntserve.py   ✅
│   │       ├── concurrent_initialization.py ✅
│   │       ├── request_migration.py   ✅
│   │       ├── no_handle.py           ✅
│   │       ├── only_ondemand.py       ✅
│   │       ├── warmup.py             ✅
│   │       └── show_events.py  ✅
│   └── online/
│       ├── scenario_A/
│       │   ├── shuntserve.py   ✅
│       │   ├── concurrent_initialization.py ✅
│       │   ├── request_migration.py   ✅
│       │   ├── no_handle.py           ✅
│       │   ├── only_ondemand.py       ✅
│       │   ├── warmup.py             ✅
│       │   └── show_events.py  ✅
│       └── scenario_B/
│           ├── shuntserve.py   ✅
│           ├── concurrent_initialization.py ✅
│           ├── request_migration.py   ✅
│           ├── no_handle.py           ✅
│           ├── only_ondemand.py       ✅
│           ├── warmup.py             ✅
│           └── show_events.py  ✅
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
8. **병렬 파이프라인 재생성**: `stop_nodes()` + `create_pipeline()` 전략(request_migration, no_handle)에서 영향받는 파이프라인을 `asyncio.gather()`로 병렬 재생성 (legacy 패턴과 동일)
9. **Percentiles**: `[1, 5, 10, 25, 50, 75, 90, 95, 99]` — P1, P5, P95 포함
10. **시간 파라미터 변수화**: `START_TIME_MIN`, `END_TIME_MIN`, `MAX_DURATION_MIN` 상수로 추출 (분 단위 정의, 사용 시 `* 60` 초 변환). 나중에 값 조절 용이.
11. **Online vs Offline**: 동일 코드 + 6가지 차이 — `time_scale` (0.0→1.0), `benchmark_type`, `OUTPUT_PATH`, `trace_output_prefix`, print header, docstring

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

### VNode.py — tensor store cleanup 강화
- `stop_tensor_store()`: TCP shutdown 실패 시 SSH `pkill -9 -f 'tensor_store'` force-kill fallback 추가
- `assert_tensor_store_stopped()`: 새 메서드 — SSH `pgrep -f 'tensor_store'`로 프로세스 잔존 확인 후 assert
- `switch_node()` / `switch_nodes()`: `stop_tensor_store()` 후 `assert_tensor_store_stopped()` 호출 추가
- Ray worker cleanup (`get_ray_stop_command()`)은 보류 (주석 처리 상태 유지)

## Completed

### UnitTest8B (Llama-3.1-8B-Instruct, 2x g6.xlarge)
- [x] `UnitTest8B/pipelines_8b.json` — 1 pipeline, 2 stages (16+16 layers), pp=[1,1]
- [x] `UnitTest8B/nodes.json` — 4 nodes (spot×2 + on_demand×2), IP 채워서 사용
- [x] `UnitTest8B/spot_trace_events.json` — 4 events: t=3 interrupt, t=6 restore+interrupt, t=9 restore
- [x] `UnitTest8B/shuntserve.py` — migration + switch_nodes
- [x] `UnitTest8B/concurrent_initialization.py` — re-routing + switch_nodes
- [x] `UnitTest8B/request_migration.py` — migration + stop_nodes/create_pipeline (병렬 재생성)
- [x] `UnitTest8B/no_handle.py` — re-routing + stop_nodes/create_pipeline (병렬 재생성)
- [x] `UnitTest8B/only_ondemand.py` — baseline, 이벤트 없음
- [x] `UnitTest8B/show_events.py` — 이벤트 매핑 디버깅 도구

### llama3-70b/offline
- [x] `scenario_{A,B}/shuntserve.py` — migration + switch_nodes
- [x] `scenario_{A,B}/concurrent_initialization.py` — re-routing + switch_nodes
- [x] `scenario_{A,B}/request_migration.py` — migration + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/no_handle.py` — re-routing + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/only_ondemand.py` — baseline, 이벤트 없음
- [x] `scenario_{A,B}/warmup.py` — 모든 노드 provision, ShuntServe pipelines + EXTRA_PIPELINES inline
- [x] `scenario_{A,B}/show_events.py` — 이벤트 매핑 디버깅 도구

### qwen3-32b/offline
- [x] `scenario_{A,B}/shuntserve.py` — migration + switch_nodes
- [x] `scenario_{A,B}/concurrent_initialization.py` — re-routing + switch_nodes
- [x] `scenario_{A,B}/request_migration.py` — migration + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/no_handle.py` — re-routing + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/only_ondemand.py` — baseline, 이벤트 없음
- [x] `scenario_{A,B}/warmup.py` — 모든 노드 provision, ShuntServe pipelines + EXTRA_PIPELINES inline
- [x] `scenario_{A,B}/show_events.py` — 이벤트 매핑 디버깅 도구

### GlobalServer
- [x] `max_duration` 지원 (`evaluation_utils.py`, `benchmark_utils.py`)
- [x] VNode.py tensor store cleanup 강화 (force-kill + assert)

### llama3-70b/online
- [x] `scenario_{A,B}/shuntserve.py` — migration + switch_nodes (time_scale=1.0)
- [x] `scenario_{A,B}/concurrent_initialization.py` — re-routing + switch_nodes
- [x] `scenario_{A,B}/request_migration.py` — migration + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/no_handle.py` — re-routing + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/only_ondemand.py` — baseline, 이벤트 없음
- [x] `scenario_{A,B}/warmup.py` — 모든 노드 provision, ShuntServe pipelines + EXTRA_PIPELINES inline
- [x] `scenario_{A,B}/show_events.py` — 이벤트 매핑 디버깅 도구

### qwen3-32b/online
- [x] `scenario_{A,B}/shuntserve.py` — migration + switch_nodes (time_scale=1.0)
- [x] `scenario_{A,B}/concurrent_initialization.py` — re-routing + switch_nodes
- [x] `scenario_{A,B}/request_migration.py` — migration + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/no_handle.py` — re-routing + stop_nodes/create_pipeline (병렬 재생성)
- [x] `scenario_{A,B}/only_ondemand.py` — baseline, 이벤트 없음
- [x] `scenario_{A,B}/warmup.py` — 모든 노드 provision, ShuntServe pipelines + EXTRA_PIPELINES inline
- [x] `scenario_{A,B}/show_events.py` — 이벤트 매핑 디버깅 도구
