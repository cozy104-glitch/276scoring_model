"""
모델별 종합 비교 실험
- Phase 2 최적 피처 세트 (13개) 사용
- 동일 CV (Stratified 5-Fold x 3) 조건
- 평가: AUC, PR-AUC, KS, Lift, Std
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from scipy.stats.mstats import winsorize

# ============================================================
# Phase 2 최적 데이터 재현
# ============================================================
GOLDEN_PATH = r"C:\Users\cozy1\Documents\276_Scoring_Model\03_flowscore_ml_ver2\outputs\AUS_v2_Golden_Training_Set.csv"
df_raw = pd.read_csv(GOLDEN_PATH, dtype={"COMPANY_ID_NORM": str})

leakage_cols = ["MORATORIUM_COUNT", "MORATORIUM_OVERDUE_AMOUNT",
                "ACCOUNT_SUSPENSION_COUNT", "CARD_ACCOUNT_COUNT",
                "NEGATIVE_COMMENT_COUNT"]
id_cols = ["COMPANY_ID", "COMPANY_ID_NORM", "TARGET_Y"]
exclude_cols = id_cols + leakage_cols
y = df_raw["TARGET_Y"]
raw_features = [c for c in df_raw.columns if c not in exclude_cols]

df_orig = df_raw[raw_features].apply(pd.to_numeric, errors="coerce")
df_orig = df_orig.fillna(df_orig.median())

df_num = df_orig.copy()
skewed_cols = ["SALES_REVENUE", "EMPLOYEE_COUNT", "CASH_RATIO", "DEBT_RATIO",
               "receivable_Total_Amt", "BNPL_Avg_Amt", "INTEREST_COVERAGE_RATIO"]
for col in df_num.columns:
    df_num[col] = winsorize(df_num[col], limits=[0.01, 0.01])
for col in skewed_cols:
    if col in df_num.columns:
        df_num[col] = np.log1p(np.abs(df_num[col])) * np.sign(df_num[col])

core    = ["CASH_RATIO", "EMPLOYEE_COUNT", "NET_PROFIT_MARGIN"]
neutral = ["receivable_Total_Amt", "BNPL_Success_Rate", "INTEREST_COVERAGE_RATIO",
           "REP_CHANGE_COUNT", "GROSS_PROFIT_MARGIN", "SALES_REVENUE"]
X = df_num[core + neutral].copy()
X["FE_GROWTH_QUALITY"]    = (df_orig["SALES_GROWTH_RATE"] * df_orig["OPERATING_MARGIN"]).replace([np.inf, -np.inf], 0).fillna(0)
X["FE_DEBT_SERVICE"]      = (df_orig["CASH_RATIO"] * (df_orig["INTEREST_COVERAGE_RATIO"] + 1)).replace([np.inf, -np.inf], 0).fillna(0)
X["FE_PROFIT_EFFICIENCY"] = ((df_orig["SALES_REVENUE"] * df_orig["OPERATING_MARGIN"] / 100) / (df_orig["EMPLOYEE_COUNT"] + 1)).replace([np.inf, -np.inf], 0).fillna(0)
X["FE_ZERO_COUNT"]        = (df_orig == 0).sum(axis=1).astype(float)

X = X.astype(float)
print(f"Data: {X.shape[1]} features, {y.sum()} positives / {len(y)} total ({y.mean()*100:.2f}%)\n")

# ============================================================
# 모델 정의
# ============================================================
POS_WEIGHT = (len(y) - y.sum()) / y.sum()  # ~25.5

models = {
    # --- Tree-based ---
    "RF (v4.9 base)": RandomForestClassifier(
        n_estimators=700, max_depth=None, min_samples_split=10,
        max_features="log2", class_weight="balanced_subsample", random_state=42, n_jobs=-1),

    "RF (depth=8)": RandomForestClassifier(
        n_estimators=500, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),

    "GBM": GradientBoostingClassifier(
        n_estimators=604, max_depth=4, learning_rate=0.022,
        subsample=0.616, min_samples_split=13, min_samples_leaf=7, random_state=42),

    # --- Boosting ---
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=POS_WEIGHT, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=-1),

    "LightGBM": LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=POS_WEIGHT, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, random_state=42, verbose=-1, n_jobs=-1),

    "CatBoost": CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=42, verbose=0),

    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200, learning_rate=0.5, random_state=42),

    # --- Imbalanced-learn ---
    "BalancedRF": BalancedRandomForestClassifier(
        n_estimators=500, max_depth=None, max_features="log2",
        random_state=42, n_jobs=-1),

    "EasyEnsemble": EasyEnsembleClassifier(
        n_estimators=30, random_state=42, n_jobs=-1),

    # --- Linear ---
    "LogisticReg (L2)": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=0.1,
                                   max_iter=1000, random_state=42))]),

    "LogisticReg (L1)": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=0.01,
                                   penalty="l1", solver="saga",
                                   max_iter=1000, random_state=42))]),

    # --- SVM ---
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale"),
            cv=3))]),

    # --- Stacking ---
    "Stacking (RF+XGB+LGB)": StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=500, max_depth=None,
                        class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        scale_pos_weight=POS_WEIGHT, random_state=42, verbosity=0)),
            ("lgb", LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        scale_pos_weight=POS_WEIGHT, random_state=42, verbose=-1)),
        ],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=5, stack_method="predict_proba", n_jobs=-1),
}

# ============================================================
# 공정 비교: Stratified 5-Fold x 3 Repeats
# ============================================================
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

def evaluate_model(name, model, X, y, cv):
    pr_aucs, ks_stats, lifts, briers, aucs = [], [], [], [], []
    for tr_idx, te_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        try:
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]
        except Exception as e:
            print(f"  [{name}] Error: {e}")
            return None
        aucs.append(roc_auc_score(y_te, y_prob))
        pr_aucs.append(average_precision_score(y_te, y_prob))
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        ks_stats.append(np.max(tpr - fpr))
        briers.append(brier_score_loss(y_te, y_prob))
        n_top = max(1, int(len(y_te) * 0.1))
        top_idx = np.argsort(y_prob)[::-1][:n_top]
        br = y_te.mean()
        lifts.append(y_te.iloc[top_idx].mean() / br if br > 0 else 0)
    return {
        "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs),
        "PR_AUC": np.mean(pr_aucs), "KS": np.mean(ks_stats),
        "Lift10": np.mean(lifts), "Brier": np.mean(briers),
    }

# ============================================================
# 실행
# ============================================================
print("=" * 90)
print(f"{'Model':<28} {'AUC':>7} {'±Std':>7} {'PR-AUC':>8} {'KS':>7} {'Lift10':>8} {'Brier':>8}")
print("=" * 90)

results = {}
for name, model in models.items():
    print(f"  Running: {name}...", end="", flush=True)
    r = evaluate_model(name, model, X, y, rskf)
    if r:
        results[name] = r
        print(f"\r  {name:<28} {r['AUC_mean']:>7.4f} {r['AUC_std']:>7.4f} "
              f"{r['PR_AUC']:>8.4f} {r['KS']:>7.4f} {r['Lift10']:>8.2f}x {r['Brier']:>8.4f}")
    else:
        print(f"\r  {name:<28} FAILED")

# ============================================================
# 최종 정렬 출력
# ============================================================
print("\n" + "=" * 90)
print("FINAL RANKING (by AUC)")
print("=" * 90)
sorted_results = sorted(results.items(), key=lambda x: x[1]["AUC_mean"], reverse=True)
for rank, (name, r) in enumerate(sorted_results, 1):
    bar = "★" * min(5, round(r["AUC_mean"] * 10 - 7))
    print(f"  #{rank:2d} {name:<28} AUC={r['AUC_mean']:.4f}±{r['AUC_std']:.4f} "
          f"KS={r['KS']:.4f} Lift={r['Lift10']:.2f}x  {bar}")

# ============================================================
# 카테고리별 최강자
# ============================================================
print("\n" + "=" * 90)
print("CATEGORY WINNER")
print("=" * 90)
best_auc  = max(results.items(), key=lambda x: x[1]["AUC_mean"])
best_std  = min(results.items(), key=lambda x: x[1]["AUC_std"])
best_lift = max(results.items(), key=lambda x: x[1]["Lift10"])
best_ks   = max(results.items(), key=lambda x: x[1]["KS"])
best_pr   = max(results.items(), key=lambda x: x[1]["PR_AUC"])

print(f"  Best AUC:    {best_auc[0]:<28} {best_auc[1]['AUC_mean']:.4f}")
print(f"  Most Stable: {best_std[0]:<28} Std={best_std[1]['AUC_std']:.4f}")
print(f"  Best KS:     {best_ks[0]:<28} {best_ks[1]['KS']:.4f}")
print(f"  Best PR-AUC: {best_pr[0]:<28} {best_pr[1]['PR_AUC']:.4f}")
print(f"  Best Lift:   {best_lift[0]:<28} {best_lift[1]['Lift10']:.2f}x")

# ============================================================
# Devil's Advocate: 과연 최고인가?
# ============================================================
best_name, best_r = sorted_results[0]
print(f"\n{'='*90}")
print("DEVIL'S ADVOCATE VERDICT")
print("=" * 90)
print(f"""
Best model: {best_name}
  AUC: {best_r['AUC_mean']:.4f} ± {best_r['AUC_std']:.4f}
  95% CI: [{best_r['AUC_mean'] - 1.96*best_r['AUC_std']:.4f}, {best_r['AUC_mean'] + 1.96*best_r['AUC_std']:.4f}]

Top 3 모델 간 AUC 차이:
  #{1} {sorted_results[0][0]}: {sorted_results[0][1]['AUC_mean']:.4f}
  #{2} {sorted_results[1][0]}: {sorted_results[1][1]['AUC_mean']:.4f}  (차이: {sorted_results[0][1]['AUC_mean']-sorted_results[1][1]['AUC_mean']:+.4f})
  #{3} {sorted_results[2][0]}: {sorted_results[2][1]['AUC_mean']:.4f}  (차이: {sorted_results[0][1]['AUC_mean']-sorted_results[2][1]['AUC_mean']:+.4f})

CV Std = {best_r['AUC_std']:.4f} → 95% CI 폭 = {1.96*best_r['AUC_std']*2:.4f}
결론: 상위 모델들의 차이가 CV 노이즈 범위 내라면 사실상 동등.
""")
