# Greedy vs. Joint 파이프라인 추출 (Reviewer R2#11)

> **R2#11:** *"The step where iteratively extract pipelines is critical but unspecified.
> Has greedy extraction been compared against jointly optimizing all K pipelines
> (for K=2, joint optimization is tractable)?"*
>
> (반복적으로 파이프라인을 추출하는 단계가 핵심인데 명시가 안 돼 있다. greedy 추출을 **K개 파이프라인을
> 동시에(jointly) 최적화하는 것**과 비교해 봤는가? K=2면 joint 최적화가 충분히 다룰 만하다.)

## 현재(배포된) 옵티마이저: greedy

ShuntServe 옵티마이저는 파이프라인을 **greedy**하게 추출한다. beam-search DP
([`shuntserve_optimizer.py`](../../../../ModelPlacement/shuntserve_optimizer.py))를 **전체 클러스터**에 대해
한 번 돌려 **효율(`soft_slo`: throughput/cost에 SLO 페널티) 기준 1등 파이프라인 하나**만 남기고, 그
파이프라인이 쓴 노드를 빼낸 뒤, 클러스터가 소진될 때까지 이를 반복한다. 그 결과 파이프라인 개수 **K**는
가변적으로 결정된다.

## 우리가 brute-force한 joint 대안

파이프라인 개수 `p`를 **고정**한 상태에서, 클러스터 노드를 **정확히 `p`개의 비어있지 않은 그룹**으로 나누는
**모든 파티션**을 enumerate → 각 그룹마다 **동일한** per-pipeline 옵티마이저로 파이프라인 1개를 뽑고 →
**total(합산) throughput**으로 모든 파티션을 랭킹한다. greedy 해(解)는 그중 특정 파티션 하나이므로, 전체
랭킹에서 greedy가 **몇 위**인지를 본다.

- 클러스터: `g5.12xlarge×2, g6.12xlarge×3, g6e.xlarge×4` (9 노드, 24 GPU)
- 그룹은 노드 multiset `(n_g5, n_g6, n_g6e)`로 표현 → 비어있지 않은 유니크 그룹은 `3·4·5 − 1 = 59`개
- **Memoization:** 한 그룹의 최적 파이프라인은 (모델 config, 그 그룹의 노드 multiset, 가격)에만 의존한다.
  서로 다른 노드 집합 위의 파이프라인은 (분석 모델상) 상호작용하지 않으므로, 유니크 그룹마다 **딱 한 번**
  최적화해 모든 파티션·모든 `p`에서 재사용한다. `p=4`까지 tractable한 이유가 바로 이것이다.
- per-pipeline 선택 기준은 **두 가지**를 모두 기록:
  - **`soft_slo`** — 논문 greedy 선택과 **동일** (주(主) 비교, 사과 대 사과)
  - **`only_throughput`** — 효율 기준에 이의를 제기할 경우를 대비한 **순수 throughput 상한**

### 파일
- 엔진: [`joint_p_common.py`](joint_p_common.py)
- 드라이버: 각 모델 `joint-p/`에 `shuntserve-p={1,2,3,4}.py` (예:
  [`qwen3-32b/joint-p/shuntserve-p=4.py`](qwen3-32b/joint-p/shuntserve-p=4.py)). 전역 최적 확인용
  p=1..9 sweep은 같은 memo 캐시로 [`joint_p_common.py`](joint_p_common.py)에서 바로 돌릴 수 있다.
- 출력: 각 드라이버가 `…/joint-p/results/joint_p=<p>_<model>.json` (+ 공유 `memo_<model>.json`)을 기록
- 실행/재현: 아래 **재현 방법** 참조.

## 재현 방법 (Reproduce)

GPU 불필요(분석 estimator). Python 환경에 `torch` + `transformers`만 있으면 된다. 모델 config는 주변 환경에서
로드한다 — 네트워크 + `hf auth login`(또는 로컬 HF 캐시). Llama-3.1-70B는 gated라 `hf auth login` 되어 있거나
캐시돼 있어야 하고, Qwen3-32B는 open이라 자동 다운로드된다. 캐시만 쓰려면(오프라인) `HF_HUB_OFFLINE=1`을
직접 export하면 된다.

```bash
cd <repo>/ArtifactEvaluation/ModelPlacement/optimizer/joint-comparison
PY=python3                       # 또는 venv 인터프리터:  PY=~/.venv/bin/python
mkdir -p logs

# (1) 메인: greedy vs joint(memoization + subset-DP) 시간 비교 + "joint 전역최적 == greedy" 검증
#     self-contained(memo를 매번 fresh 계산). 워커=코어수-1, 인자로 워커 수 지정 가능: `joint_optimum_dp.py 31`
nohup $PY joint_optimum_dp.py > logs/joint_dp_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/joint_dp_*.log            # 진행 확인 (로그 끝의 SUMMARY 블록이 결과)

# (2) per-p 전체 파티션 sweep + 랭킹(+ valid/feasibility 플래그). 결과 JSON은 joint-p/results/ 에 저장
#     첫 실행이 memo 캐시 생성 → 같은 모델의 다른 p는 즉시. memo 재계산은 `--refresh`.
$PY qwen3-32b/joint-p/shuntserve-p=4.py   > logs/joint_qwen_p4.log  2>&1
$PY qwen3-32b/joint-p/shuntserve-p=2.py   > logs/joint_qwen_p2.log  2>&1
$PY llama3-70b/joint-p/shuntserve-p=2.py  > logs/joint_llama_p2.log 2>&1
#   p=1, p=3 드라이버도 동일하게 존재 (shuntserve-p={1,3}.py)

# (3) p=1..9 전역 sweep / 전역최적(subset-DP)을 직접 호출하고 싶으면:
$PY -c "import joint_p_common as J; \
m=J.compute_memo('llama3-70b', cache_path='llama3-70b/joint-p/results/memo_llama3-70b.json', modes=('soft_slo',), log=lambda *a:None); \
print(J.joint_optimum_dp(m,'soft_slo'))"
```

참고:
- `hf auth login` + 네트워크면 두 모델 config가 자동 로드된다(Qwen은 open이라 자동 다운로드). 완전 오프라인으로
  돌리려면 `HF_HUB_OFFLINE=1`을 export하되 두 config가 모두 캐시돼 있어야 한다.
- 옵티마이저(`run_test_case`)와 하이퍼파라미터(`top_k=3`, `max_stages=13`, `soft_slo`)는 배포 greedy와 **동일**
  하다 — joint은 오케스트레이션(memoization + subset-DP)만 바꾼 것이고 optimizer 자체는 손대지 않았다.
- **결과 추출:** `grep -A8 SUMMARY logs/joint_dp_*.log`
- 시간 측정값은 머신 의존적이다(표에 측정 환경 표기: 로컬 8코어 / m8a.8xlarge 32코어 등).

## 파티션 경우의 수 (p개 그룹 분할) 계산법

인스턴스 **타입**이 `t`종이고 타입 `i`에 동일한 인스턴스가 `nᵢ`개 있다고 하자(여기선 `t=3`,
`n = (n_g5, n_g6, n_g6e) = (2, 3, 4)`, 총 9노드). 같은 타입의 인스턴스는 **서로 구별 불가**하므로, 한 그룹은
"타입별 개수 벡터 `(k₁,…,k_t)`"로만 특정된다. 클러스터를 `p`개 그룹으로 나누는 경우의 수는 다음 3단계로
센다.

**① 라벨 있는 그룹, 빈 그룹 허용.** 타입 `i`의 동일한 `nᵢ`개를 구별되는 `p`개 그룹에 나눠 담는 방법
(stars & bars) `= C(nᵢ + p − 1, p − 1)`. 타입마다 독립이므로

  `N_ordered(빈그룹허용) = ∏ᵢ C(nᵢ + p − 1, p − 1)`

**② 모든 그룹 비어있지 않게 (포함배제).** 그룹이 비는 사건은 타입을 가로질러 결합되므로(어떤 그룹이 비려면
*모든* 타입에서 0개를 받아야 함) 포함배제로 뺀다:

  `N_ordered(비어있지않음) = Σ_{j=0}^{p} (−1)ʲ · C(p, j) · ∏ᵢ C(nᵢ + (p−j) − 1, (p−j) − 1)`

  (`p−j = 0`이고 어떤 `nᵢ>0`이면 그 항은 0 — 양의 인스턴스를 0개 그룹에 담을 수 없음.)

**③ 라벨 없는 그룹 (우리가 쓰는 값).** 파이프라인(그룹)은 서로 **교환 가능**하므로 unordered 파티션을 센다.
이때 단순히 `p!`로 나누면 **틀린다** — 같은 그룹이 반복되는 파티션은 서로 다른 순열이 `p!`개보다 적기
때문이다. (예: `p=4`에서 `②/4! = 3756/24 = 156.5` → 비정수.) 정확한 **닫힌 공식**은 `p`개 그룹 라벨에 작용하는
대칭군 `S_p`의 궤도(orbit) 수를 세는 **Burnside 보조정리**로 주어진다:

```
U(p) = (1/p!) Σ_{σ∈S_p} Fix(σ) = Σ_{λ⊢p} F(λ) / z_λ

  λ⊢p     : p의 정수분할(= S_p의 cycle type), 길이 r 사이클이 a_r개
  z_λ      = ∏_r r^{a_r} · a_r!          (cycle type λ인 순열 수 = p!/z_λ)
  F(λ)     = Σ_{T⊆{cycles}} (−1)^{|T|} · ∏_{i=1}^{t} r_i({c_k : k∉T})   (빈 사이클 포함배제)
  r_i(D)   = [q^{n_i}] ∏_{c∈D} 1/(1−q^c)
           = #{ (x_c)_{c∈D} ≥ 0 : Σ_{c∈D} c·x_c = n_i }   (사이클 k의 박스들이 각각 c_k배 기여)
```

**특수예 `p=2`** (`S_2={e, τ}`): `e`(cycle 1+1)는 `F(e)=∏ᵢ(nᵢ+1)−2 = 60−2 = 58`(ordered non-empty),
`τ`(swap, cycle 2)는 두 박스가 같아야 하므로 `F(τ)=∏ᵢ[nᵢ 짝수]=[2][3][4]=1·0·1=0`(g6가 3개=홀수라 반으로
못 쪼갬). 따라서 `U(2)=½(58+0)=29`. → 즉 `U(2)=½[∏ᵢ(nᵢ+1)−2 + ∏ᵢ[nᵢ even]]`이고, 마지막(두 박스 동일)
항이 0이라 우연히 `58/2`가 맞은 것이다(모든 nᵢ가 짝수였다면 그 항 때문에 단순 `/2!`는 틀린다).

위 `U(p)` 공식은 아래 ③열 값(코드 enumeration)과 **모든 p에서 정확히 일치**하며, 실무 코드는 이 공식 대신
각 파티션을 **정규형(그룹 시그니처 정렬 튜플)으로 dedup**해 같은 값을 센다(파티션 리스트 자체가 필요하므로).

클러스터 `n=(2,3,4)`에 대한 실제 수치 (③이 실험에서 enumerate한 파티션 수):

| p | ① 빈그룹 허용 `∏C(nᵢ+p−1,p−1)` | ② 비어있지않음(포함배제) | ③ **unordered (사용값)** |
|---|---|---|---|
| 1 | 1 | 1 | **1** |
| 2 | 60 | 58 | **29** |
| 3 | 900 | 723 | **126** |
| 4 | 7,000 | 3,756 | **191** |
| 5 | 36,750 | 10,155 | **151** |
| 6 | 148,176 | 15,570 | **75** |
| 7 | 493,920 | 13,685 | **26** |
| 8 | 1,425,600 | 6,440 | **6** |
| 9 | 3,675,375 | 1,260 | **1** |

(모든 `p`에 대한 unordered 파티션 합 = 606.) 코드 구현
([`enumerate_partitions`](joint_p_common.py))은 정확히 이 절차다: 타입별 compositions(`_compositions`) ×
Cartesian product(`itertools.product`)로 라벨 있는 분배를 만들고 → 빈 그룹이 있는 것을 버리고 → 정렬 튜플로
정규화해 `set`으로 dedup → 개수를 센다.

**memoization 비용은 `p`와 무관**하다: 최적화해야 할 **유니크 그룹**의 수는 클러스터의 비어있지 않은
부분-multiset 개수 `= ∏ᵢ(nᵢ+1) − 1 = 3·4·5 − 1 = 59`로, 모든 `p`가 이 59개를 공유한다(그래서 한 번 캐싱하면
다른 `p`는 즉시 랭킹만). 반면 ①의 값이 보여주듯 라벨 있는 분배 수 `∏ᵢ C(nᵢ+p−1, p−1)`는 노드/타입 수가
늘면 급격히 커진다 — 전수 brute-force가 대규모 클러스터로 확장되지 않는 이유다.

## Baseline 검증

현재 코드로 greedy를 다시 돌리면 논문의 예측 throughput을 **정확히 재현**한다
(`ReferenceData/.../offline_shuntserve.json` → `predicted_total_throughput_rps`):

| 모델 | greedy K | greedy total (이번 실행) | 논문 `predicted_total_throughput_rps` |
|---|---|---|---|
| Llama-3.1-70B | 2 | **2.8305** | 2.8305 |
| Qwen3-32B | 4 | **9.4241** | 9.4241 |

memo로 재계산한 greedy total은 논문값 및 per-pipeline throughput과 `<1e-6`로 일치한다 → 각 greedy
파이프라인이 해당 그룹의 단독 최적해와 정확히 같다는 의미이며, memoization의 건전성을 입증한다.

> 참고: 리포지토리에 들어있던 `optimizer/results/.../predicted_*.json`(total 2.382 / 1.517)은 **구버전(stale)**
> 결과다. 이번 작업에서는 **건드리지 않고 원래대로 복원**했으며, 모든 비교는 논문 regime과 일치하는 현재
> 코드로 수행했다.

## 결과: p-sweep으로 본 joint 최적해

joint 최적화 = "클러스터 노드를 여러 파이프라인으로 나누는 **최선의 분할**을 찾기"이고, 파이프라인 개수
자체도 최적화 대상이다. 이를 brute-force하기 위해 **p = 파이프라인 개수**를 `1..9`로 sweep하며 각 p에서
최선의 파티션을 구하고, **전체 sweep의 최댓값**이 전역 joint 최적해다. (주 기준 `soft_slo` = 논문과 동일,
total throughput(req/s)으로 랭킹.)

> **"1 그룹 = 1 파이프라인"은 일반성을 잃지 않는다.** 리뷰어가 말한 "K개를 동시에 뽑아 joint 최적화"는,
> 클러스터를 K개 그룹으로 명시적으로 나누고 각 그룹 안에서 layer partitioning(DP)으로 파이프라인 1개씩
> 뽑는 것과 **동치**다. 어떤 영역을 2개 이상으로 쪼개는 게 유리하면 그건 **그룹 수가 더 많은 파티션(더 큰
> p)** 으로 이미 enumeration에 들어있다. p=1..9를 전부 훑으므로 가능한 모든 파이프라인 집합(개수 무관)이
> 빠짐없이 커버되며, 따라서 어떤 "그룹핑 후 추출" 전략도 우리 enumeration이 지배한다.

각 p에서의 joint-최적 total throughput (soft_slo). **`best`** = 임의 분할의 최댓값(일부 그룹이 메모리상
infeasible이면 그 그룹은 0 기여 = 노드 idle), **`(valid)`** = *p개 파이프라인이 전부 valid*한 구성만의
최댓값(`—` = 그런 구성 자체가 불가):

| p (파이프라인 수) | 파티션 수 | Llama-3.1-70B  best (valid) | Qwen3-32B  best (valid) |
|---|---|---|---|
| 1 | 1 | 1.995 (1.995) | 5.865 (5.865) |
| **2** | 29 | **2.831 (2.831)** ← K | 8.493 (8.493) |
| 3 | 126 | 2.501 (2.125) | 8.959 (8.959) |
| **4** | 191 | 2.177 **(—)** | **9.424 (9.424)** ← K |
| 5 | 151 | 1.877 (—) | 9.138 (9.138) |
| 6 | 75 | 1.590 (—) | 8.956 (8.956) |
| 7 | 26 | 1.250 (—) | 6.717 (5.395) |
| 8 | 6 | 0.437 (—) | 4.243 (—) |
| 9 | 1 | 0.000 (—) | 3.091 (—) |
| **valid 파이프라인 최대 수** | | **3** | **7** |
| **전역 최적 (valid 한정)** | | **2.8305 @ p=2** | **9.4241 @ p=4** |
| **greedy (배포 해)** | | **2.8305 (K=2)** | **9.4241 (K=4)** |

> **Llama-3.1-70B는 메모리 때문에 valid 파이프라인이 최대 3개**다 — 1~2노드 같은 작은 그룹엔 70B가 안
> 올라간다. 그래서 **p≥4에선 *p개 모두 valid*한 구성이 없고(`—`)**, 표의 `best(any)` p≥3 값은 사실
> **2개만 valid + 나머지 그룹은 infeasible(노드 idle)** 인 부분해다(예: Llama p=3 best = `{g6e×1}✗(0) +
> {g6×3,g6e×3}(2.161) + {g5×2}(0.340)`). Qwen3-32B는 더 작아 valid 최대 7개. **모든 파이프라인이 valid한
> 경우로 한정해도** 전역 최적은 그대로 greedy다 (Llama 2.8305 @ p=2, Qwen 9.4241 @ p=4).

두 모델 모두 (valid 구성 기준) total throughput이 **`p = K`(greedy가 적응적으로 고른 개수)에서 peak**를
찍고, 바로 그 지점에서 greedy의 파티션이 전수 joint 최적해와 **정확히 일치**한다 — Llama p=2에서 29개 중
**#1**, Qwen p=4에서 191개 중 **#1**이며 `soft_slo`·`only_throughput` 양쪽에서 동일하다.

**전역 joint 최적해 = greedy.** greedy는 파이프라인 개수 K도, 그 K개로의 노드 분할도 모두 최적으로 골랐다.
joint 최적화는 어떤 p에서도 greedy를 이기지 못한다 — 단지 같은 답을 **훨씬 더 비싸게** 찾을 뿐이다(아래
"Algorithm time" 참조).

## Algorithm time: greedy vs joint (memoization + subset-DP)

joint은 greedy와 **동일한 최적해**를 주지만 **훨씬 비싸다**. optimizer(`run_test_case`)는 손대지 않고,
두 가지 **외부 최적화**만 적용해 joint을 최대한 싸게 만들었다:

- **memoization:** 유니크 그룹(`∏(n_i+1)−1 = `**59개**)만 한 번씩 옵티마이저 호출(병렬 가능). 파티션이
  13.8억개여도 참조 값의 종류는 59개뿐.
- **subset-DP:** 파티션을 enumerate하지 않고 `best[S]=max_{∅≠T⊆S}(thr[T]+best[S−T])`로 **전역 최적**(임의
  #pipelines)을 바로 계산. 비용 `O(∏(n_i+1)(n_i+2)/2)≈900연산` → 실측 **~0.3 ms**.

[joint_optimum_dp.py](joint_optimum_dp.py)로 **m8a.8xlarge(32 vCPU, 워커 31)** 에서 측정:

| | Greedy (배포 해) | Joint (memoization + subset-DP) |
|---|---|---|
| optimizer 호출 수 | **K회** (Llama 2, Qwen 4) | **59회** (유니크 그룹, `p`와 무관) |
| subset-DP / 파티션 처리 | — | **~0.3 ms** (13.8억 파티션 안 돌고 전역최적) |
| 결과 = greedy 최적? | (기준) | ✅ Llama **2.8305** · Qwen **9.4241** 동일 |
| **wall-clock** | Llama **36.6 s** · Qwen **31.8 s** | Llama **41.1 s** · Qwen **31.3 s** |
| **직렬 CPU(opt 합)** | Llama 36.6 s · Qwen 31.8 s | Llama 497.8 s · Qwen 382.9 s |
| **greedy 대비** | 1× | **wall ~1.0–1.1× · CPU ~12–14×** |

→ subset-DP는 사실상 공짜(~0.3ms)라 **병목은 오직 59개 그룹 옵티마이저 호출**(13.8억 파티션이 아님). **32코어에선
joint wall이 greedy와 거의 같다**(Llama 41 vs 37s, Qwen 31 vs 32s) — 병렬이 59개 호출을 가려서다. 다만 **CPU(총
work)는 ~12–14× 그대로**이고, 이건 코어를 더 부어도 줄지 않는다.

**per-group 측정(`joint_optimum_dp.json`의 `per_group_times`):** 가장 느린 그룹은 전부 **7~8노드짜리 큰 그룹**
(예 `{g5×2,g6×3,g6e×4}` Llama 35s / Qwen 27s; best pipeline이 full-cluster 최적과 동일)이고 작은 그룹은
sub-second다. 즉 **joint wall의 하한 = 가장 느린 단일 그룹 시간 ≈ greedy의 full-cluster 첫 호출**이라, 코어가
충분하면 joint wall이 greedy 수준으로 수렴한다(위 32코어 결과). 반대로 타입이 늘면 그룹 수 `∏(n_i+1)`가 **지수
폭발**(large_hetero 15타입 → 32,767 그룹)해 wall을 평평히 유지하려면 `~그룹수`만큼의 코어가 필요하고, CPU/비용은
지수적으로 그대로다. greedy는 타입 수와 무관하게 ~K번. 리뷰어의 "K=2면 tractable" 단서도 이 비용 때문이며,
**배포에 greedy를 쓰는 선택이 정당하다**(같은 최적해, 훨씬 싼 비용·확장성).

### joint 내부 비용 상세 (memoization 효과)

8 CPU 코어에서 측정. 각 `p`가 건드리는 유니크 그룹들에 대한 옵티마이저 시간의 합 (soft_slo 기준).

| 실험 | 최적화한 유니크 그룹 | 옵티마이저 시간 (memoized) | 옵티마이저 시간 (naive, memo 없음) | 병렬 wall-clock¹ |
|---|---|---|---|---|
| Llama p=2 | 58 | 1148 s | 1148 s | 386 s |
| Llama p=3 | 55 | 952 s | **2663 s** (378회 호출) | 0.8 s² |
| Qwen p=2 | 58 | 796 s | 796 s | 261 s |
| Qwen p=3 | 55 | 673 s | **2060 s** (378회 호출) | 0.6 s² |
| Qwen p=4 | 49 | 477 s | **1931 s** (764회 호출) | 0.6 s² |

¹ 모델당 1회: *전체 59그룹 × 두 기준*을 병렬로 최적화하는 데 걸린 wall-clock.
² Qwen p=3·p=4 등은 p=2 실행이 만든 캐시 memo를 재사용 → 랭킹만 수행.

memoization은 per-group 작업을 `p`와 무관하게 ≤59회 옵티마이저 호출로 압축한다. `p=2`에서는 memoized = naive
(각 2-파티션의 두 그룹이 서로 다름), `p=4`에서는 약 4배 절감(유니크 그룹 49 vs. 출현 횟수 764). 다만 이건
joint **내부** 최적화일 뿐이고, 위에서 보듯 joint 전체는 여전히 greedy보다 훨씬 비싸다.

## 검증 (독립 audit 4종)

- **Memoization 건전성 — sound.** `estimator_utils.get_throughput`은 단일 파이프라인의 `node_layer_comb`
  + 모델 config만의 순수 함수이며, 가변 전역 상태나 파이프라인 간 결합이 없다 → 그룹별로 따로 최적화해
  throughput을 **합산**하는 것이 정확하다. "제한 최적해 = 전역 최적해의 부분집합" 성질은 *보장*된 결과다
  (결정론적 DP, `latency_slo`→∞이므로 `soft_slo`가 순수 throughput/cost 최대화로 환원) — memo가 논문의
  per-pipeline·total throughput을 `<1e-6`로 재현하는 것이 그 증거(우연이 아님).
- **코드 정확성 — sound.** `enumerate_partitions`를 독립 재구현해 동일한 파티션 집합(p=2/3/4 → 29/126/191,
  누락·중복 없음)을 확인했고, 랭킹·greedy 배치(canonicalization)·memoized vs naive 시간 회계도 모두 확인.
- **독립 재계산 — sound.** memo 캐시 없이 옵티마이저를 처음부터 다시 돌려 5개 그룹과 두 joint total을
  `<1e-3`로 재현(예: Qwen `{g6e×4}` 5.8652, `{g6×3}` 2.6278; Qwen p=4 total 9.4241; Llama p=2 total 2.8305).
- **적대적 리뷰어 — sound_with_caveats.** 결론은 유효. 지적은 정확성 결함이 아니라 아래의 범위/일반화
  관련 caveat에 한정됨.

## 범위 및 한계 (R2 선제 대응)

- **프레이밍.** 이것은 *조건부* 주장이다 — *"K개 파이프라인 그룹을 배치한다고 할 때, greedy의 파티션이
  모든 p=K 파티션 중 최적인가?"* → 두 검증점 모두 **그렇다**. greedy의 **적응적 K 선택**(반복 피드백을 통한)은
  별개의 추가 이점이며, Qwen p=2 행이 더 작은 K를 강제하면 엄밀히 나빠짐을 보여준다.
- **검증 규모.** 모델 2종, 9노드·인스턴스 3종 클러스터 1개, 워크로드 1개(in=763, out=232), p=1..9 전체 sweep.
  여기서 brute-force가 가능한 이유는 파티션 수가 작고(p별 1~191개) memoization이 per-group 작업을 ≤59회로 묶기
  때문이다. **전수 enumeration은 대규모 클러스터로 확장되지 않는다**(파티션 수가 조합적으로 폭증). memoization은
  per-group 재계산은 줄이지만 파티션 수 자체는 줄이지 못하므로, 대규모에서는 더 영리한 탐색이 필요하다. 본
  결과는 *평가한 규모에서* greedy를 검증한 것이지 보편적 최적성 증명은 아니다.
- **모델 가정.** 시스템 total throughput을 독립적인 per-pipeline throughput의 합으로 본다(파이프라인 간
  큐잉/라우팅/공정성 없음) — 분석 모델 및 offline 벤치마크와 일관. greedy와 joint가 *같은* per-pipeline
  기준을 쓰므로 상대 랭킹은 기준 선택에 강건하며, 결론은 `soft_slo`·`only_throughput` 양쪽에서 동일하다.

## Rebuttal 초안

**(한국어)**

> 우리는 greedy 추출을 "K개 파이프라인을 동시에 joint 최적화"하는 방식과 brute-force로 비교했다. 9노드
> 클러스터를 정확히 p개의 비어있지 않은 그룹으로 나누는 모든 파티션을 enumerate하고(파이프라인 개수 p를
> 1..9로 sweep), 각 그룹을 동일한 per-pipeline DP로 최적화한 뒤 total throughput으로 랭킹했으며, per-group
> 최적화는 59개 유니크 sub-cluster에 대해 memoize했다. total throughput은 정확히 **p = K**(greedy가 적응적으로
> 고른 값)에서 최대였고, 그 **전역 joint 최적해가 greedy 배치와 정확히 일치**한다 — Llama-3.1-70B(K=2,
> 2.83 req/s) 29개 중 1위, Qwen3-32B(K=4, 9.42 req/s) 191개 중 1위이며 효율·순수 throughput 기준 모두에서
> 그렇다. 즉 greedy는 **파티션 선택**과 **적응적 K 선택** 모두에서 전역 최적이다. 다만 이 **동일한** 최적해를
> brute-force로 찾는 데는 greedy(K번 호출, ~35 s)의 약 14~30배(수백~수천 초)의 옵티마이저 시간이 들고
> 클러스터 크기에 따라 더 폭증하므로, 배포 시스템이 greedy를 쓰는 선택은 정당하다.

**(English — paste-ready)**

> We compared greedy extraction against jointly optimizing all K pipelines by brute force. Sweeping the
> number of pipelines p from 1 to 9, we enumerate every partition of the 9-node cluster into exactly p
> non-empty groups, optimize one pipeline per group with the same per-pipeline DP, and rank by total
> throughput; per-group optimization is memoized over the 59 unique sub-clusters. Total throughput peaks
> exactly at p = K, the number greedy selects adaptively, and that **global joint optimum coincides with
> greedy's placement**: #1 of 29 for Llama-3.1-70B (K=2, 2.83 req/s) and #1 of 191 for Qwen3-32B (K=4,
> 9.42 req/s), under both an efficiency and a raw-throughput criterion. Greedy is thus optimal in both its
> partition and its choice of K. Reaching this same optimum by brute force, however, costs ~14–30× the
> optimizer time of greedy (hundreds–thousands of seconds vs. greedy's ~35 s of K calls) and blows up with
> cluster size, which is why the deployed system uses greedy.
