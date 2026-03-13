 276 SME Credit Scoring System 2026-03-13


본 프로젝트는 276 Holdings의 중소기업(SME) 대상 공급망 금융 리스크를 선제적으로 탐지하기 위해 개발된 하이브리드 신용평가 엔진(Hybrid Credit Scoring Engine)의 연구 개발 백서입니다.

기존의 정적이고 단편적인 재무제표 중심의 심사 관행을 탈피하여, ① 심사역의 직관을 데이터로 정량화하고, ② 이기종 데이터를 스키마 기반으로 통합하였으며, ③ 행동 지표(결제 타임스탬프)와 통계적 무결성(Leakage 차단, SMOTE)을 결합하여 실전 예측력(AUC 0.7063)을 확보한 3단계 기술 고도화 과정을 담고 있습니다.

System Architecture & Repository Structure

본 레포지토리는 모델의 진화 과정(Phase 1 $\rightarrow$ 3)에 따라 구조화되어 있으며, 각 폴더는 독립적인 R&D 마일스톤을 나타냅니다.

[Phase 1] 심사역 행동 모사 엔진 및 리스크 사각지대 탐색
[Phase 2] 이기종 데이터 스키마 매핑 및 Feature Mart 인프라
[Phase 3] Target Leakage 차단 및 SMOTE 검증 하이브리드 엔진


Phase 1: Baseline Audit & Human-Mimic Engine

"심사역의 직관을 정량화하여 1,514개 기업의 리스크 사각지대를 발굴하다"

기존 매뉴얼(v3)에 기반한 심사역의 행동을 모사하는 룰 기반(Rule-based) 엔진을 가동하여 인공지능 도입을 위한 기준점(Baseline)을 수립했습니다.

* 기초 자산 정밀 점검:
* 재무 데이터 3,580건 및 비재무 고용 데이터 7,881건 확보.
* 기업당 평균 대표자 변경 1.15회, 3회 이상 변경된 불안정 기업 49개사 조기 식별.
* Human-Mimic 엔진 가동 결과 (1,514개사):
* LOW(우량/승인) 1,222개사 vs HIGH(거절/고위험) 130개사 분류.
* 심사역의 부정적 코멘트 및 과거 거절 이력을 감점 요소(-20점)로 알고리즘화.
* 리스크 사각지대 발견:
* AI 변수 중요도 분석 결과, 재무 지표가 아닌 `LINKED_PARTNERS`(공급망 연결성, 15.38%)와 `SALES_GROWTH_RATE`(13.79%)가 전체 부실 위험의 42.3%를 설명함을 입증. 초경량/고정밀 심사 모델로의 전환 타당성 확보.

Phase 2: Schema Mapped Data Integration (v1.0)

"파편화된 원천 데이터를 신용평가용 Feature Mart로 자산화하다"

초기 모델의 한계를 극복하기 위해, 분산된 Snowflake 운영 DB와 외부 데이터(NICE 등)를 결합하는 강력한 데이터 엔지니어링 파이프라인을 구축했습니다.

* 스키마 정밀 매핑 (Schema Mapping):
* 수천 개의 Raw 데이터를 신용평가 표준 스키마(FR_VAL)에 정밀하게 매핑.
* 대차대조표(BS) 연산을 통해 모델의 핵심인 `CASH_RATIO`(현금비율) 직접 산출 및 결합.
* Feature Mart 4.0 완성:
* 7대 핵심 재무지표를 포함한 19개 완전체 심사 Feature 생성 완료.
* 데이터 누락(NaN)과 자본잠식 등의 예외 케이스를 완벽히 처리하여 AI 학습을 위한 견고한 인프라(Data Foundation) 구축.

Phase 3: Robust Hybrid Engine (v5.2)

"데이터 누수를 차단하고 통계적 한계를 극복한 실전형 위기 탐지 모델"

단 11개의 불량(Hard Default) 샘플이 가진 극단적 데이터 불균형(Class Imbalance)을 해결하고, 모델이 미래를 미리 아는 오류(Target Leakage)를 제거하여 실무 적용 가능한 엔진을 완성했습니다.

* 통계적 무결성 확보 (Robustness Check):
* SMOTE (가상 데이터 합성): 정상(1,503개) 대비 부족한 연체(11개) 데이터를 SMOTE 기법으로 1,503개까지 확장하여 우연성을 배제한 AI 재학습 수행.
* Target Leakage 완벽 제거: 과거 데이터로 미래를 예측하도록 시점(Point-in-Time)을 재정렬. 초기 과적합 점수(AUC 0.9925)를 실전 신뢰성을 갖춘 ROC-AUC 0.7063으로 교정.
* 행동 지표(Behavioral Data) 도입:
* 기업의 '자금 압박 절박함'을 나타내는 오후 결제 비중(`FE_PM_RATIO_ACTUAL`) 및 `FE_LIQUIDITY_STRESS`를 변수로 투입하여 변별력 극대화.
* 최종 타겟 선별 (Top 10 High-Risk):
* `CASH_RATIO`(15.08%), `FE_LIQUIDITY_STRESS`(8.98%) 중심의 26개 핵심 피처를 기반으로 1,514개 기업 전수 스코어링.
* ID `0000000072`, `0000000557` 등 재무적으로 드러나지 않던 부도 확률(ML_PD_RATE) 76.06%의 초고위험 10개사 정밀 타격 및 취급 제한.

Target Variable (Y) Definition Strategy

본 모델의 성공은 단순한 부도 이력을 넘어, '사전 위기 징후'를 타겟팅한 Y값 설계에 있습니다.

1. Hard Default: 연체 기록(11개, 0.7%), 휴폐업 이력(1,418개)
2. Soft Default: 심사 거절 이력 및 평가 로그 스캔
3. Synthesis & Calibration: SMOTE를 통한 소수 클래스 증강 및 `CalibratedClassifierCV`를 활용한 예측 확률(PD)의 실무적 보정. 이를 통해 1건 이상의 리스크가 중첩된 잠재 부실군을 정밀하게 학습.


Author: 임지훈

Date: 2026. 03. 13.
