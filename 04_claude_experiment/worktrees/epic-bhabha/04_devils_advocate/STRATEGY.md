# Devil's Advocate Model Improvement Strategy
# 악마의 변호인 모델 고도화 전략서

> **원칙**: 모든 결정에 "정말 이것이 최선인가?"를 묻고, 실험으로 증명한다.
> **현재 최고 AUC**: 0.7347 (v4.9, 단일 80/20 분할, random_state=42)

---

## Phase 0: 현재 모델의 근본적 문제 진단

### 0-1. 가장 먼저 챌린지할 것: "0.7347이라는 숫자를 믿을 수 있는가?"

**현재 문제점:**
- `random_state=42` 고정된 단일 80/20 분할 → 이 분할이 운 좋은 분할일 수 있음
- 양성 57건 중 테스트에 ~11건만 배정 → 1건의 정답/오답이 AUC를 ±0.05 이상 흔듦
- v3.0→v4.9까지 모든 실험이 **동일한 test set**으로 평가 → test set에 과적합 가능

**Devil's Advocate 질문:**
> "random_state를 42가 아닌 다른 값으로 바꾸면 AUC가 0.65일 수도 있지 않은가?"

**실험 E0-1: Baseline 신뢰구간 측정**
```
방법: Stratified 5-Fold CV × 3회 반복 (총 15번 평가)
기대: AUC 평균 ± 표준편차 → 예: 0.72 ± 0.06
판정: std > 0.05면 "현재 0.7347은 노이즈에 가까움"으로 간주
```

### 0-2. Target 변수 자체를 챌린지

**현재 문제점:**
- Y=1이 57건 (3.76%) — 연체 11건(OVERDUE) + 모라토리엄 위험 46건
- 모라토리엄 기준: MORATORIUM_OVERDUE_AMOUNT > 0 OR ACCOUNT_SUSPENSION > 0 OR WORKOUT > 0
- 이 정의가 "진짜 부실"을 의미하는가?

**Devil's Advocate 질문:**
> "모라토리엄 연체액 500만원인 기업과 1억인 기업을 같은 Y=1로 묶는 것이 맞는가?"
> "연체 이력이 있지만 이미 해소된 기업도 Y=1인가?"

**실험 E0-2: Target 정의 민감도 분석**
```
Target_A: 연체(OVERDUE)만 (11건) → 너무 적어서 학습 불가할 수 있음
Target_B: 현재 정의 (57건) → 기존
Target_C: 모라토리엄 등록액 > 0까지 확대 (96건)
Target_D: Continuous target (연체 금액 회귀) → AUC 대신 RMSE
각각 5-Fold CV로 비교 → 어떤 정의가 가장 안정적인 판별력을 보이는가?
```

---

## Phase 1: 전처리 (Preprocessing) 챌린지

### 1-1. "fillna(0)이 정말 맞는가?"

**현재 코드 문제:**
```python
# feature_engineering.ipynb
df[col] = df[col].fillna(df[col].median())  # 일부 컬럼
df.fillna(0, inplace=True)  # 나머지 전부 0
```

**Devil's Advocate 질문:**
> "BNPL_Req_Count=0은 'BNPL을 한 번도 안 썼다'인가, '데이터가 없다'인가?"
> "321/1514 기업만 BNPL 데이터 보유 → 나머지 1,193개의 0은 거짓 정보 아닌가?"

**실험 E1-1: 결측 처리 전략 비교**
```
Strategy_A: 현재 (fillna(0)) → Baseline
Strategy_B: fillna(median) 전체 적용
Strategy_C: fillna(0) + 결측 indicator 컬럼 추가
            → BNPL_IS_MISSING, RECEIVABLE_IS_MISSING 등
Strategy_D: 결측 비율 70% 이상 컬럼 제거 + 나머지 median
평가: 5-Fold CV AUC로 비교. 가장 높은 전략 채택.
챌린지: "Strategy_C가 이겼다면, indicator 자체가 leakage는 아닌지 확인"
```

### 1-2. "이상치를 방치하는 것이 맞는가?"

**현재 데이터 상태:**
```
DEBT_RATIO:  max=909,090 (중앙값 70.16) → 12,950배
CASH_RATIO:  max=158,280 (중앙값 12.84) → 12,326배
SALES_REVENUE: max=209B (중앙값 2.04M) → 102,476배
OPERATING_MARGIN: 평균 -73.36 (중앙값 0.44) → 극단적 좌편향
```

**Devil's Advocate 질문:**
> "DEBT_RATIO가 909,090인 기업 하나가 tree split을 지배하고 있지 않은가?"
> "RF는 이상치에 강건하다고 하지만, 57건의 양성 중 이상치가 포함되면 패턴이 왜곡된다"

**실험 E1-2: 이상치 처리 전략 비교**
```
Strategy_A: 현재 (방치) → Baseline
Strategy_B: Winsorization (1%-99% 절단)
Strategy_C: Winsorization (5%-95% 절단)
Strategy_D: Robust Scaler (중앙값 + IQR 기반)
Strategy_E: Log변환 (현재 SALES_REVENUE만 → 전 컬럼 확대)
평가: 5-Fold CV AUC
챌린지: "B가 이겼다면, 1% cutoff가 최적인지 0.5%, 2%도 비교"
```

### 1-3. "DEBT_RATIO가 object 타입인 이유는?"

**현재 문제:**
```
# feature_engineering.ipynb 진단 결과
DEBT_RATIO: Type=object, NaN Count=116 (pd.to_numeric 후)
```

**Devil's Advocate 질문:**
> "116개의 변환 실패값은 무엇인가? 문자열? 특수기호? 이 116건이 Y=1과 연관되어 있다면?"

**실험 E1-3: DEBT_RATIO 정밀 진단**
```
Step 1: 변환 실패 116건의 원본값 확인
Step 2: 이 116건 중 Y=1 비율 확인 (전체 3.76% 대비)
Step 3: 변환 실패 자체를 indicator 피처로 추가
챌린지: "변환 실패=부실 기업의 재무제표 미제출일 수 있다 → 매우 강한 신호"
```

---

## Phase 2: 피처 엔지니어링 (Feature Engineering) 챌린지

### 2-1. "현재 피처들이 정말 독립적인 기여를 하는가?"

**현재 문제:**
```
피처 28개 중 다수가 동일 원천에서 파생:
- CASH_RATIO, FE_LIQUIDITY_STRESS, Z_CASH_RATIO → 전부 현금 기반
- EMPLOYEE_COUNT, FE_LOG_EMPLOYEE → 동일 원천
- SALES_REVENUE, FE_LOG_REVENUE, FE_NET_DEPENDENCY → 매출 기반
```

**Devil's Advocate 질문:**
> "Z_CASH_RATIO와 CASH_RATIO를 둘 다 넣으면 RF가 이 둘을 번갈아 split해서
>  중요도가 분산(dilution)되는 것 아닌가?"

**실험 E2-1: 다중공선성 vs 판별력 트레이드오프**
```
Strategy_A: 현재 28개 전부 → Baseline
Strategy_B: VIF > 10 피처 제거 (다중공선성 기준)
Strategy_C: 그룹별 1개만 남기기
            Liquidity: CASH_RATIO만
            Size: FE_LOG_EMPLOYEE만
            Revenue: SALES_REVENUE만
            등
Strategy_D: PCA로 차원 축소 (95% 분산 설명)
Strategy_E: Permutation Importance 기반 선택 (상위 15개)
평가: 5-Fold CV AUC
챌린지: "피처를 줄였는데 성능이 같다면, 원래 피처가 노이즈였다는 뜻"
```

### 2-2. "결측 패턴 자체가 피처가 될 수 있는가?"

**현재 상황:**
- BNPL 데이터: 321/1514 기업만 보유 (21%)
- PM_RATIO: 거래 로그가 있는 기업만 보유
- 행동 데이터가 없는 기업 ≈ 플랫폼 비활성 기업 → 리스크 신호?

**실험 E2-2: 메타 피처 생성**
```python
# 각 기업의 "데이터 존재 여부" 자체를 피처로
FE_HAS_BNPL = (BNPL_Req_Count > 0).astype(int)
FE_HAS_RECEIVABLE = (receivable_Total_Amt > 0).astype(int)
FE_HAS_PM_RATIO = (PM_RATIO_PAYABLE_TRANSACTIONS > 0).astype(int)
FE_MISSING_COUNT = 각 기업의 0값 컬럼 수
FE_DATA_COMPLETENESS = 비제로 컬럼 수 / 전체 컬럼 수
```
```
평가: 메타 피처 추가 전후 5-Fold CV AUC 비교
챌린지: "HAS_BNPL이 Y와 상관관계가 높다면,
        이것은 '예측 피처'인가 '선택 편향'인가?"
        → v4.5에서 has_log가 leakage였던 전례 반드시 참고
```

### 2-3. "파생 피처의 비즈니스 로직이 맞는가?"

**현재 파생 피처 챌린지:**

| 피처 | 수식 | 챌린지 |
|------|------|--------|
| FE_LIQUIDITY_STRESS | (DEBT_RATIO+1)/(CASH_RATIO+1) | DEBT_RATIO max=909K → 이 피처도 극단값. +1이 충분한 보정인가? |
| FE_NET_DEPENDENCY | SALES_REVENUE/(LINKED_PARTNERS+1) | LINKED_PARTNERS 범위 0~657. 657개 거래처는 대기업인데, 이걸 "의존도"로 해석? |
| FE_PROFIT_EFFICIENCY | (REVENUE×MARGIN)/EMPLOYEE | MARGIN이 음수이면 음의 효율 → 의미가 뒤집힘 |
| FE_CASH_VELOCITY | CASH/log(REVENUE) | REVENUE=0이면 log(0+1)=0 → 결국 CASH_RATIO의 변형 |
| FE_LIQUIDITY_DEAD_CROSS | CASH×(1-PM_RATIO) | PM_RATIO 없는 기업은 0으로 채워져 있어서 CASH와 동일 |

**실험 E2-3: 피처별 순수 기여도 측정**
```
방법: Leave-One-Feature-Out (LOFO) Importance
각 피처를 하나씩 빼고 5-Fold CV AUC 측정
→ AUC가 떨어지지 않으면 그 피처는 불필요
→ AUC가 올라가면 그 피처는 오히려 해로움
챌린지: "기여도가 0에 가까운 피처가 5개 이상이면,
        현재 피처 엔지니어링 전략 자체를 재검토"
```

### 2-4. "Rank 기반 피처가 더 강건하지 않은가?"

**Devil's Advocate 질문:**
> "Z-score는 이상치에 민감하다. 평균과 표준편차가 극단값에 끌리니까.
>  차라리 순위(rank)를 쓰면 분포 가정 없이 상대적 위치를 표현할 수 있지 않은가?"

**실험 E2-4: Rank vs Z-score**
```python
# Rank-based features
FE_CASH_RANK = CASH_RATIO.rank(pct=True)
FE_DEBT_RANK = DEBT_RATIO.rank(pct=True)
FE_REVENUE_RANK = SALES_REVENUE.rank(pct=True)
```
```
평가: Z-score 피처 대신 Rank 피처를 넣고 5-Fold CV AUC 비교
챌린지: "Rank는 모든 값을 0~1로 압축하므로 정보 손실이 있을 수 있다"
→ 두 가지를 동시에 넣으면? (Z + Rank 혼합)
```

### 2-5. "새로운 교차 피처 후보"

**현재 없지만 비즈니스 로직상 유의미한 피처:**
```python
# 수익성 있는 성장인가? (적자 성장 vs 흑자 성장 구분)
FE_GROWTH_QUALITY = SALES_GROWTH_RATE * OPERATING_MARGIN

# 이자 상환 여력 (현금 × 이자보상배율)
FE_DEBT_SERVICE = CASH_RATIO * INTEREST_COVERAGE_RATIO

# 업력 대비 규모 (있다면)
FE_MATURITY_INDEX = SALES_REVENUE / (COMPANY_AGE + 1)

# 거래처 다양성 대비 매출 집중도
FE_REVENUE_PER_PARTNER = SALES_REVENUE / (LINKED_PARTNERS + 1)  # 이미 FE_NET_DEPENDENCY와 유사
```
```
평가: 각 후보를 하나씩 추가하면서 5-Fold CV AUC 변화 관찰
챌린지: "교차 피처를 추가할수록 과적합 리스크 증가 →
        Occam's Razor 원칙으로 최소한의 피처 세트를 추구"
```

---

## Phase 3: 모델링 (Modeling) 챌린지

### 3-1. "Random Forest가 최선인가?"

**현재 실험 이력:**
```
RF baseline:     0.7063
Soft Voting:     0.7265
Stacking RF+XGB: 0.7340
Calibrated RF:   0.7347
```

**Devil's Advocate 질문:**
> "RF 계열만 시도했다. 데이터가 1,514건으로 작으면
>  gradient boosting이나 regularized linear model이 더 나을 수 있다"

**실험 E3-1: 모델 종류별 공정 비교**
```
모든 모델을 동일한 5-Fold CV, 동일한 피처 세트로 비교:

Model_A: RandomForest (현재 최적 설정)
Model_B: XGBoost (scale_pos_weight 튜닝)
Model_C: LightGBM (min_child_samples 튜닝 → 소규모 데이터 대응)
Model_D: CatBoost (auto class balancing)
Model_E: LogisticRegression + WoE 변환 (해석 가능성 확보)
Model_F: Balanced Bagging + Decision Tree
Model_G: SVM (RBF kernel, class_weight='balanced')

평가: 5-Fold CV AUC 평균 ± std
챌린지: "최고 AUC 모델이 아니라, 가장 안정적인(std가 작은) 모델이 더 중요할 수 있다"
```

### 3-2. "class_weight='balanced'가 최적인가?"

**현재 접근:**
- `class_weight='balanced'` → 소수 클래스에 ~25배 가중치
- `class_weight='balanced_subsample'` → v4.9에서 승리

**Devil's Advocate 질문:**
> "25배가 맞는가? 15배나 40배가 더 나을 수 있지 않은가?"

**실험 E3-2: 불균형 처리 전략 비교**
```
Strategy_A: class_weight='balanced' (~25:1)
Strategy_B: class_weight='balanced_subsample'
Strategy_C: scale_pos_weight 수동 튜닝 (10, 15, 20, 25, 30, 40)
Strategy_D: Focal Loss (gamma=1, 2, 3)
Strategy_E: BalancedRandomForest (imblearn)
Strategy_F: EasyEnsemble (imblearn)
Strategy_G: Cost-sensitive threshold tuning (predict_proba 임계값 최적화)
평가: 5-Fold CV에서 AUC + PR-AUC 동시 비교
챌린지: "AUC는 높은데 PR-AUC가 낮다면, 실제 운용시 정밀도가 낮다"
```

### 3-3. "max_depth=None이 정말 최적인가?"

**v4.9 결과:**
- 최적 설정: `max_depth=None` (무제한)
- 1,514건, 양성 57건에서 무제한 깊이 → 과적합 가능성 높음

**Devil's Advocate 질문:**
> "RandomizedSearchCV n_iter=10은 너무 적다.
>  깊이 5, 7, 9, 11, None만 시도했는데, 6이나 8이 더 나을 수 있다"

**실험 E3-3: 정밀 하이퍼파라미터 탐색**
```
도구: Optuna (100+ trials, Bayesian optimization)
탐색 공간:
  n_estimators: [200, 2000]
  max_depth: [3, 20] + None
  min_samples_split: [2, 30]
  min_samples_leaf: [1, 20]
  max_features: ['sqrt', 'log2', 0.3, 0.5, 0.7]
  class_weight: ['balanced', 'balanced_subsample']
  + custom weight dict

목적함수: 5-Fold CV AUC (반드시 CV 안에서 평가)
챌린지: "Optuna가 찾은 최적값이 overfitting이 아닌지
        → nested CV (outer 5-fold × inner 5-fold)로 검증"
```

### 3-4. "단일 모델 vs 앙상블, 어디까지 가는 것이 합리적인가?"

**현재 시도:**
- Soft Voting: 0.7265
- Stacking RF+XGB: 0.7340

**Devil's Advocate 질문:**
> "Stacking이 Calibrated RF(0.7347)보다 낮았다.
>  앙상블이 항상 좋은 것은 아니다. 왜 이 경우에 단일 모델이 이겼는가?"

**실험 E3-4: 앙상블 심화 실험**
```
Ensemble_A: Stacking (RF + XGB + LightGBM, meta=LR) → 다양성 증가
Ensemble_B: Blending (holdout 기반 메타 학습)
Ensemble_C: Weighted Average (각 모델 CV AUC 비례 가중)
Ensemble_D: 2단계 하이브리드
            Stage1: 규칙 기반 (CASH=0 → 자동 HIGH_RISK)
            Stage2: ML 모델 (나머지 기업만 학습)
평가: 5-Fold CV AUC
챌린지: "앙상블 복잡도 대비 개선폭이 0.01 미만이면,
        단일 모델을 선택하고 해석가능성을 확보하는 것이 실무적으로 나을 수 있다"
```

---

## Phase 4: 평가 체계 (Evaluation) 챌린지

### 4-1. "AUC만으로 충분한가?"

**Devil's Advocate 질문:**
> "AUC 0.73이면 금융권 기준으로 어느 수준인가?
>  실제로 상위 10%를 잡았을 때 부실 기업을 몇 개나 포착하는가?"

**추가 평가 지표:**
```
1. PR-AUC (Precision-Recall AUC)
   → 불균형 데이터에서 AUC보다 민감

2. KS Statistic
   → 금융권 표준. KS > 0.3이면 실전 투입 가능

3. Top-K Lift
   → 상위 10% 스코어 기업 중 실제 부실 비율 / 전체 부실 비율
   → Lift > 5이면 실용적

4. Expected Calibration Error (ECE)
   → 예측 확률이 실제 빈도와 얼마나 일치하는가

5. Brier Score Decomposition
   → Reliability + Resolution + Uncertainty 분해
```

### 4-2. "교정(Calibration)이 제대로 되었는가?"

**현재:** Isotonic Calibration (CV=3)

**Devil's Advocate 질문:**
> "Isotonic은 비모수적이라 데이터가 적으면 overfitting된다.
>  57건의 양성으로 isotonic 곡선을 안정적으로 추정할 수 있는가?"

**실험 E4-2: 교정 전략 비교**
```
Cal_A: Isotonic (현재, CV=3)
Cal_B: Isotonic (CV=5)
Cal_C: Platt Scaling (Logistic)
Cal_D: Temperature Scaling
Cal_E: 교정 없음 (raw probability)
평가: ECE + Brier Score + Calibration Curve 시각화
챌린지: "교정 후 AUC가 변하지 않아야 한다 (교정은 순서를 바꾸지 않으므로)"
```

---

## Phase 5: 실행 순서 및 게이트 기준

### 전체 실행 로드맵

```
[Gate 0] 신뢰구간 측정 → "현재 모델의 진짜 실력을 안다"
    │
    ├── E0-1: 5-Fold CV 신뢰구간
    ├── E0-2: Target 정의 민감도
    │
    │ GATE 판정: CV AUC 평균이 0.70 이상이면 Phase 1로
    │            0.70 미만이면 Target 재정의부터
    ▼
[Gate 1] 전처리 최적화 → "깨끗한 입력 데이터를 확보한다"
    │
    ├── E1-1: 결측 처리 전략
    ├── E1-2: 이상치 처리 전략
    ├── E1-3: DEBT_RATIO 정밀 진단
    │
    │ GATE 판정: Phase 0 대비 AUC 개선 확인.
    │            개선 없으면 "전처리는 현행 유지" 결론
    ▼
[Gate 2] 피처 엔지니어링 → "모델에게 더 좋은 재료를 준다"
    │
    ├── E2-1: 다중공선성 정리
    ├── E2-2: 메타 피처 (결측 패턴)
    ├── E2-3: LOFO 중요도 검증
    ├── E2-4: Rank vs Z-score
    ├── E2-5: 신규 교차 피처
    │
    │ GATE 판정: Gate 1 대비 AUC 개선 확인.
    │            개선 없으면 "피처는 현행 유지" 결론
    ▼
[Gate 3] 모델링 최적화 → "최적의 알고리즘을 찾는다"
    │
    ├── E3-1: 모델 종류 비교
    ├── E3-2: 불균형 처리 전략
    ├── E3-3: Optuna 하이퍼파라미터
    ├── E3-4: 앙상블 심화
    │
    │ GATE 판정: Gate 2 대비 AUC 개선 확인.
    │            단일 모델이 앙상블과 0.01 이내면 단일 모델 채택
    ▼
[Gate 4] 평가 고도화 → "결과를 신뢰할 수 있게 만든다"
    │
    ├── E4-1: 다중 메트릭 평가
    ├── E4-2: 교정 전략 비교
    │
    └── 최종 산출물: 고도화된 모델 + 성능 리포트
```

### 각 실험의 기록 양식

모든 실험은 아래 형식으로 기록한다:

```markdown
## 실험 E{phase}-{number}: {실험명}

### 가설
- 현재 방식의 문제점: ...
- 제안하는 대안: ...
- 예상 결과: ...

### 실험 설계
- 비교 대상: A vs B vs C
- 평가 방법: Stratified 5-Fold CV
- 평가 지표: AUC (primary), PR-AUC (secondary)

### 결과
| 전략 | AUC 평균 | AUC std | PR-AUC | 비고 |
|------|---------|---------|--------|------|
| A    |         |         |        |      |
| B    |         |         |        |      |

### Devil's Advocate 챌린지
- 이 결과가 우연이 아닌 근거: ...
- 남은 의문점: ...
- 다음 실험에서 확인할 것: ...

### 결론
- 채택/기각: ...
- 이유: ...
```

---

## 핵심 원칙 요약

1. **모든 비교는 5-Fold CV로** — 단일 분할 결과는 증거로 인정하지 않음
2. **한 번에 하나만 변경** — 전처리+피처+모델을 동시에 바꾸면 원인 추적 불가
3. **개선 없으면 현행 유지** — 복잡성을 늘리는 것은 개선이 증명될 때만
4. **모든 실험 기록** — 실패한 실험도 기록 (같은 실수 반복 방지)
5. **Leakage 상시 경계** — 새 피처 추가 시 반드시 "이것이 미래 정보가 아닌가?" 확인
