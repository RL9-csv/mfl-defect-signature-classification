# MFL Defect Signature Classification
**누설자속탐상(MFL) 결함 신호의 공간 패턴 분류 — 장비 임계룰 너머의 failure signature 식별**

장비가 진폭 임계로 "결함 유무"를 판정한 *그 다음 단계*를 다룬다. 결함 스파이크의 **공간 signature(형태·방향)** 를 분류해, 단순 양·불 판정이 아닌 failure mode 후보를 식별함. 반도체 Product Engineering의 wafer/bin map failure-pattern 분석과 구조적으로 유사한 문제로 정의함.

---

## 1. 문제 정의

MFL 검사 장비는 진폭 임계(REJECT H: 1.5V / 26mm)로 결함 이벤트를 자가판정함. 따라서 **"결함 유무" 분류는 임계 한 줄(trivial)** 이라 ML로 풀 가치가 낮음. 문제를 두 단계로 재정의함:

- **1차** — 임계를 넘은 결함 후보(component) 중 **구조적 결함 후보(structured) vs 고립 노이즈성 artifact(isolated)** 분리
- **2차** — 구조적 결함의 **형태 분류**: `staircase`(대각 진행) / `band`(원주방향) / `line`(길이방향). 형태가 결함 원인(root cause)의 단서가 됨

핵심: structured와 isolated, 그리고 세 형태는 **진폭이 모두 임계를 넘음** → 진폭(임계)으로는 못 가름. 오직 **공간 패턴**으로만 갈림 → 2D CNN의 필요성이 데이터로 성립함.

## 2. 데이터

- 강철바 MFL 비파괴검사. 86일 / 878 LOT / 약 85,000 bar
- 각 bar = 길이방향 스캔(약 507 point) × 10채널 (A1–A5 상단, B1–B5 하단)
- 라벨: 장비 자가판정(임계) 출발 → 형태 규칙 기반 **weak label** 설계 (정답 라벨 아님)

## 3. 방법

| 항목 | 내용 |
|---|---|
| 입력 | component 패치 `(4, 64, 10)` = [log1p 진폭, 임계 mask, 길이방향 gradient, 채널방향 gradient] |
| 센서 topology | 약 30만 component의 채널 전이 빈도 분석으로 인접관계 **역추정** → 원주순서 `A1–…–A5–B1–…–B5` (A5–B1 인접 확인, 양끝 미연결 → replicate padding) |
| 모델 | 2D CNN (3× Conv–BN–GELU + Dropout, class-weighted, cosine LR, label smoothing) |
| 비교군 | Gradient Boosting (shape feature) baseline |
| 검증 | **LOT 단위 group split** (인접 window 누수 방지), 지표: PR-AUC / macro-F1 |

## 4. 결과

| 실험 | baseline (GB) | 2D CNN |
|---|---|---|
| 1차 (구조 vs 노이즈) | PR-AUC 0.859 | **PR-AUC 0.961** |
| 2차 (형태 3-class) | macro-F1 0.748 | **macro-F1 0.903** |

**Ablation (2차):**
- **채널 셔플** — CNN 0.894 → **0.650** (−0.244). 채널 순서를 깨면 GB(0.748)보다도 낮아짐 → **센서 공간 배치·방향성이 형태분류의 결정적 동력**임을 입증
- **late-fusion(명시적 방향벡터 concat)** — 0.894 → 0.651 하락 → CNN이 raw에서 공간 방향을 이미 충분히 학습함을 확인, 불필요 피처 제거 (Occam)
- **데이터·gradient 채널** — 데이터 5배 + gradient 2채널이 2차 macro-F1을 0.728 → 0.903으로 견인

## 5. 핵심 발견

1. 진폭 임계로 못 가르는 결함을 **공간 패턴(CNN)** 으로 분리 → 2D CNN 정당성을 데이터로 입증
2. 형태분류에서 **채널 셔플 −0.24 폭락** = 센서 물리 배치가 결정적
3. late-fusion / LOT 메타데이터는 효과 없음을 **실증** → 도구를 맹신하지 않고 단순 구조 채택

## 6. 한계 (정직)

- weak label(형태 규칙 기반) → 성능 천장이 규칙 정확도에 묶임
- 진짜 결함의 물리적 ground truth(절단검사)는 없음 → "진짜 결함" 단정이 아닌 "후보"로 표현
- 향후: 약 85,000 bar **무라벨** 신호로 self-supervised 사전학습 → 적은 라벨로 fine-tune

## 7. 반도체 Product Engineering 연결

- 결함 **형태(signature) → failure mode → 의심 원인 좁히기** = yield-learning 루프
- 장비 임계가 만든 **허위경보(overkill) 분리** = PE 핵심 KPI(수율 손실 방지)
- wafer/bin map의 공간 failure-pattern classification과 **구조적으로 유사** (물리현상 동일시 아님)

## Stack
Python · PyTorch · scikit-learn · SciPy · NumPy
