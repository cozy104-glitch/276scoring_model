# Devil's Advocate 실험 결과 보고서

## 전체 파이프라인 성능 추이

| Phase | AUC Mean | AUC Std | 개선폭 | 핵심 변경 |
|-------|---------|---------|--------|----------|
| **0: Baseline** | 0.7554 | 0.0541 | - | 현재 모델 진짜 실력 측정 |
| **1: Preprocessing** | 0.7593 | 0.0547 | +0.0039 | median + Winsor1% + Log |
| **2: Features** | **0.7890** | **0.0292** | **+0.0297** | 21개→13개 정제 + 신규 4개 |
| **3: Modeling** | 0.7890 | 0.0292 | +0.0000 | RF가 이미 최적 (Optuna로도 개선 불가) |

**총 개선: AUC +0.0336, Std -46% (0.054 → 0.029)**

---

## Phase 0: Baseline Challenge 결과

### E0-1: 기존 AUC 0.7347은 신뢰할 수 있는가?
- **아니요.** random_state=42 단일 분할의 AUC는 **0.7007**이었음 (기존 보고와 다름)
- 10개 random_state 테스트: **0.6669 ~ 0.8213** (0.15 범위)
- CV 진짜 실력: **0.7554 ± 0.0541**
- **결론:** 단일 분할 결과는 신뢰 불가. 모든 후속 실험에서 5-Fold CV × 3 반복 사용

### E0-2: Target 정의 민감도
| Target | 양성 수 | AUC | Std |
|--------|---------|-----|-----|
| A: 현재 (57건) | 57 | **0.7554** | 0.0541 |
| B: 엄격 (49건) | 49 | 0.7291 | 0.0764 |
| C: 확대 (74건) | 74 | 0.7377 | 0.0545 |

- **결론:** 현재 Target A가 최적. 재정의 불필요.

### E0-3: 다면 평가
| 지표 | Phase 0 | Phase 최종 | 변화 |
|------|---------|-----------|------|
| ROC-AUC | 0.7554 | **0.7890** | +0.0336 |
| KS Statistic | 0.5584 | **0.5856** | +0.0272 |
| PR-AUC | 0.0996 | 0.0978 | -0.0018 |
| Top-10% Lift | 2.00x | 1.89x | -0.11x |
| Brier Score | 0.0599 | 0.0676 | +0.0077 |

---

## Phase 1: Preprocessing Challenge 결과

### E1-1: 결측 처리 전략
| 전략 | AUC | Std |
|------|-----|-----|
| A: fillna(0) - 현행 | 0.7569 | 0.0541 |
| **B: fillna(median)** | **0.7574** | 0.0533 |
| C: 0 + indicator | 0.7490 | 0.0549 |
| D: median + indicator | 0.7467 | 0.0522 |

- **핵심 발견:** BNPL 데이터는 Y=1 기업 **100%가 0값** (데이터 미보유)
- indicator 피처가 오히려 성능을 떨어뜨림 → 선택 편향 주입 위험

### E1-2: 이상치 처리 전략
| 전략 | AUC | Std |
|------|-----|-----|
| A: 방치 | 0.7574 | 0.0533 |
| B: Winsor 1-99% | 0.7586 | 0.0525 |
| **F: Winsor + Log** | **0.7586** | **0.0520** |

- **결론:** Winsor 1-99% + Log 변환이 최고 AUC + 최고 안정성

### E1-3: DEBT_RATIO 진단
- 변환 실패 116건의 원본값: **"자본잠식"** (한국어 문자열)
- 변환 실패 그룹 부실률: **11.21%** (전체 3.76%의 **2.98배**)
- indicator 추가 효과: +0.0007 (미미하지만 채택)

---

## Phase 2: Feature Engineering Challenge 결과 (가장 큰 성과)

### E2-1: LOFO (Leave-One-Feature-Out) 분석

**충격적 발견: 21개 피처 중 12개가 HARMFUL (제거하면 성능이 오름)**

| 분류 | 피처 수 | 피처 목록 |
|------|---------|----------|
| **USEFUL** (빼면 하락) | 3 | CASH_RATIO (-0.033), EMPLOYEE_COUNT (-0.039), NET_PROFIT_MARGIN (-0.013) |
| NEUTRAL | 6 | receivable_Total_Amt, BNPL_Success_Rate, INTEREST_COVERAGE_RATIO, REP_CHANGE_COUNT, GROSS_PROFIT_MARGIN, SALES_REVENUE |
| **HARMFUL** (빼면 상승) | 12 | LINKED_PARTNERS (+0.012), DEBT_RATIO (+0.007), OPERATING_MARGIN (+0.007), UPLOADED_FILE_COUNT (+0.006) 등 |

### E2-2: 신규 피처 효과
| 피처 | 효과 | 판정 |
|------|------|------|
| FE_GROWTH_QUALITY (성장률×이익률) | **+0.0121** | **GAIN** |
| FE_PROFIT_EFFICIENCY (인당 이익) | **+0.0086** | **GAIN** |
| FE_ZERO_COUNT (결측 수) | **+0.0054** | **GAIN** |
| FE_DEBT_SERVICE (현금×이자보상) | +0.0026 | GAIN |
| FE_CASH_RANK | -0.0005 | FLAT |
| FE_LIQUIDITY_STRESS | -0.0032 | LOSS |
| FE_NET_DEPENDENCY | -0.0040 | LOSS |

### 최종 피처 세트 비교
| 세트 | 피처 수 | AUC | Std |
|------|---------|-----|-----|
| A: 핵심 3개만 | 3 | 0.7769 | 0.0366 |
| B: 핵심+중립 9개 | 9 | 0.7656 | 0.0298 |
| C: 전체 21개 (Phase 1) | 21 | 0.7593 | 0.0547 |
| **D: 최적 13개** | **13** | **0.7890** | **0.0292** |
| E: 전체 신규 포함 19개 | 19 | 0.7765 | 0.0310 |

---

## Phase 3: Modeling Challenge 결과

### Optuna 최적화 (각 50 trials)
| 모델 | Inner CV AUC | Outer CV AUC | Std |
|------|-------------|-------------|-----|
| RF-Optuna | 0.7952 | 0.7862 | 0.0266 |
| XGB-Optuna | 0.7954 | 0.7762 | 0.0358 |
| GBM-Optuna | **0.8077** | 0.7880 | 0.0328 |
| Stacking (RF+XGB+GBM) | - | 0.7797 | 0.0265 |
| **RF-Phase2-base** | - | **0.7890** | **0.0292** |

**Devil's Advocate 결론:**
- Optuna로 150회 탐색해도 기본 RF 설정을 이기지 못함
- GBM이 inner CV에서 0.8077로 높았지만, outer CV에서 하락 → 과적합 시사
- **RF(n=700, depth=None, balanced_subsample)가 이 데이터에 가장 적합**
- 앙상블(Stacking)이 단일 모델보다 낮음 → 57건 양성에서 복잡한 모델은 오히려 해로움

---

## 핵심 발견 요약

### 1. 피처 정리가 가장 큰 레버
- 21개 → 13개로 줄이면서 AUC +0.03, Std -46%
- **"피처를 추가하는 것이 아니라 제거하는 것이 개선"**

### 2. 핵심 신호는 3개
- **CASH_RATIO**: 가장 강력한 단일 예측 인자 (LOFO: -0.033)
- **EMPLOYEE_COUNT**: 기업 규모의 프록시 (LOFO: -0.039)
- **NET_PROFIT_MARGIN**: 수익성 (LOFO: -0.013)

### 3. "자본잠식" = 숨겨진 강력 신호
- DEBT_RATIO 컬럼에 "자본잠식"이라는 한국어 텍스트가 116건 존재
- 이 기업들의 부실률은 전체의 **2.98배** (11.21% vs 3.76%)

### 4. BNPL/행동 데이터는 현재 무의미
- Y=1 기업 57건 중 **0건**이 BNPL 데이터 보유
- 이 피처들은 모두 노이즈로 작용 (LOFO에서 HARMFUL 판정)

### 5. 모델 복잡도 ≠ 성능
- Optuna 150 trials로도 기본 RF를 이기지 못함
- Stacking이 단일 RF보다 낮음
- 57건의 양성 샘플에서는 단순한 모델이 더 안정적

---

## 최종 추천 파이프라인

```
[전처리]
1. pd.to_numeric(errors='coerce')
2. fillna(median)
3. Winsorize 1%-99%
4. Log transform (SALES_REVENUE, EMPLOYEE_COUNT, CASH_RATIO, DEBT_RATIO,
                   receivable_Total_Amt, BNPL_Avg_Amt, INTEREST_COVERAGE_RATIO)

[피처 13개]
Core: CASH_RATIO, EMPLOYEE_COUNT, NET_PROFIT_MARGIN
Neutral: receivable_Total_Amt, BNPL_Success_Rate, INTEREST_COVERAGE_RATIO,
         REP_CHANGE_COUNT, GROSS_PROFIT_MARGIN, SALES_REVENUE
Engineered: FE_GROWTH_QUALITY, FE_DEBT_SERVICE, FE_PROFIT_EFFICIENCY, FE_ZERO_COUNT

[모델]
RandomForestClassifier(
    n_estimators=700, max_depth=None, min_samples_split=10,
    max_features='log2', class_weight='balanced_subsample', random_state=42
)

[성능]
ROC-AUC:      0.7890 ± 0.0292
KS Statistic: 0.5856 ± 0.0595
PR-AUC:       0.0978 ± 0.0147
```
