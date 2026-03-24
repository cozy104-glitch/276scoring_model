import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier
from scipy.stats.mstats import winsorize

print("=" * 70)
print("PHASE 3: MODELING CHALLENGE (Optuna + Ensemble)")
print("=" * 70)

# ============================================================
# Phase 2 최적 데이터 재현
# ============================================================
GOLDEN_PATH = r"C:\Users\cozy1\Documents\276_Scoring_Model\03_flowscore_ml_ver2\outputs\AUS_v2_Golden_Training_Set.csv"
df_raw = pd.read_csv(GOLDEN_PATH, dtype={"COMPANY_ID_NORM": str})
leakage_cols = ["MORATORIUM_COUNT", "MORATORIUM_OVERDUE_AMOUNT", "ACCOUNT_SUSPENSION_COUNT", "CARD_ACCOUNT_COUNT", "NEGATIVE_COMMENT_COUNT"]
id_cols = ["COMPANY_ID", "COMPANY_ID_NORM", "TARGET_Y"]
exclude_cols = id_cols + leakage_cols
y = df_raw["TARGET_Y"]
raw_features = [c for c in df_raw.columns if c not in exclude_cols]

df_orig = df_raw[raw_features].apply(pd.to_numeric, errors="coerce")
df_orig = df_orig.fillna(df_orig.median())

df_num = df_orig.copy()
skewed_cols = ["SALES_REVENUE", "EMPLOYEE_COUNT", "CASH_RATIO", "DEBT_RATIO", "receivable_Total_Amt", "BNPL_Avg_Amt", "INTEREST_COVERAGE_RATIO"]
for col in df_num.columns:
    df_num[col] = winsorize(df_num[col], limits=[0.01, 0.01])
for col in skewed_cols:
    if col in df_num.columns:
        df_num[col] = np.log1p(np.abs(df_num[col])) * np.sign(df_num[col])

core = ["CASH_RATIO", "EMPLOYEE_COUNT", "NET_PROFIT_MARGIN"]
neutral = ["receivable_Total_Amt", "BNPL_Success_Rate", "INTEREST_COVERAGE_RATIO", "REP_CHANGE_COUNT", "GROSS_PROFIT_MARGIN", "SALES_REVENUE"]
X = df_num[core + neutral].copy()
X["FE_GROWTH_QUALITY"] = (df_orig["SALES_GROWTH_RATE"] * df_orig["OPERATING_MARGIN"]).replace([np.inf, -np.inf], 0).fillna(0)
X["FE_DEBT_SERVICE"] = (df_orig["CASH_RATIO"] * (df_orig["INTEREST_COVERAGE_RATIO"] + 1)).replace([np.inf, -np.inf], 0).fillna(0)
X["FE_PROFIT_EFFICIENCY"] = ((df_orig["SALES_REVENUE"] * df_orig["OPERATING_MARGIN"] / 100) / (df_orig["EMPLOYEE_COUNT"] + 1)).replace([np.inf, -np.inf], 0).fillna(0)
X["FE_ZERO_COUNT"] = (df_orig == 0).sum(axis=1).astype(float)

print(f"Features: {X.shape[1]}, Target: {y.sum()} pos / {len(y)} total")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# ============================================================
# Optuna: RF (50 trials)
# ============================================================
print("\n[RF Optimization - 50 trials]")
def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": 42, "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    return scores.mean()

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(rf_objective, n_trials=50)
print(f"  Best RF AUC (inner CV): {study_rf.best_value:.4f}")
print(f"  Params: {study_rf.best_params}")

# ============================================================
# Optuna: XGBoost (50 trials)
# ============================================================
print("\n[XGBoost Optimization - 50 trials]")
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42, "verbosity": 0
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(xgb_objective, n_trials=50)
print(f"  Best XGB AUC (inner CV): {study_xgb.best_value:.4f}")
print(f"  Params: {study_xgb.best_params}")

# ============================================================
# Optuna: GBM (50 trials)
# ============================================================
print("\n[GBM Optimization - 50 trials]")
def gbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "random_state": 42,
    }
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study_gbm = optuna.create_study(direction="maximize")
study_gbm.optimize(gbm_objective, n_trials=50)
print(f"  Best GBM AUC (inner CV): {study_gbm.best_value:.4f}")
print(f"  Params: {study_gbm.best_params}")

# ============================================================
# Final Comparison: 5-Fold CV x 3 Repeats
# ============================================================
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON (5-Fold CV x 3)")
print("=" * 70)

best_rf_model = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
best_xgb_model = XGBClassifier(**study_xgb.best_params, random_state=42, verbosity=0)
best_gbm_model = GradientBoostingClassifier(**study_gbm.best_params, random_state=42)

# Stacking ensemble
stack_model = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(**study_xgb.best_params, random_state=42, verbosity=0)),
        ("gbm", GradientBoostingClassifier(**study_gbm.best_params, random_state=42)),
    ],
    final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
    cv=5, stack_method="predict_proba", n_jobs=-1
)

all_models = {
    "RF-Optuna": best_rf_model,
    "XGB-Optuna": best_xgb_model,
    "GBM-Optuna": best_gbm_model,
    "Stacking": stack_model,
    "RF-Phase2-base": RandomForestClassifier(n_estimators=700, max_depth=None, min_samples_split=10, class_weight="balanced_subsample", random_state=42),
}

final_results = []
for name, model in all_models.items():
    nj = -1 if "Stacking" not in name else 1
    aucs = cross_val_score(model, X, y, cv=rskf, scoring="roc_auc", n_jobs=nj)
    final_results.append((name, aucs.mean(), aucs.std()))
    print(f"  {name:20s}: AUC={aucs.mean():.4f} +/- {aucs.std():.4f}")

# ============================================================
# Phase 4: Multi-metric evaluation on best model
# ============================================================
final_results.sort(key=lambda x: x[1], reverse=True)
print(f"\nBEST: {final_results[0][0]} (AUC={final_results[0][1]:.4f})")
best_name = final_results[0][0]
best_model = all_models[best_name]

print("\n" + "=" * 70)
print(f"PHASE 4: MULTI-METRIC EVALUATION ({best_name})")
print("=" * 70)

pr_aucs, ks_stats, lifts_10, brier_scores = [], [], [], []
for train_idx, test_idx in rskf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    best_model.fit(X_tr, y_tr)
    y_prob = best_model.predict_proba(X_te)[:, 1]

    pr_aucs.append(average_precision_score(y_te, y_prob))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ks_stats.append(np.max(tpr - fpr))
    brier_scores.append(brier_score_loss(y_te, y_prob))

    n_top = max(1, int(len(y_te) * 0.1))
    top_idx = np.argsort(y_prob)[::-1][:n_top]
    top_rate = y_te.iloc[top_idx].mean()
    base_rate = y_te.mean()
    lifts_10.append(top_rate / base_rate if base_rate > 0 else 0)

print(f"""
[Multi-Metric Results - {best_name}]
{"="*50}
ROC-AUC:        {final_results[0][1]:.4f} +/- {final_results[0][2]:.4f}
PR-AUC:         {np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}
KS Statistic:   {np.mean(ks_stats):.4f} +/- {np.std(ks_stats):.4f}
Brier Score:    {np.mean(brier_scores):.4f} +/- {np.std(brier_scores):.4f}
Top-10% Lift:   {np.mean(lifts_10):.2f}x +/- {np.std(lifts_10):.2f}x
{"="*50}
""")

# ============================================================
# OVERALL SUMMARY
# ============================================================
print("=" * 70)
print("DEVIL'S ADVOCATE: FULL PIPELINE SUMMARY")
print("=" * 70)
print(f"""
Phase 0 Baseline:     AUC=0.7554 +/- 0.0541 (21 features, fillna(0))
Phase 1 Preprocess:   AUC=0.7593 +/- 0.0547 (+0.0039, median+Winsor+Log)
Phase 2 Features:     AUC=0.7890 +/- 0.0292 (+0.0297, 13 optimized features)
Phase 3 Modeling:     AUC={final_results[0][1]:.4f} +/- {final_results[0][2]:.4f} ({final_results[0][0]})

Total improvement:    +{final_results[0][1] - 0.7554:.4f} AUC
Stability improvement: {0.0541:.4f} -> {final_results[0][2]:.4f} std

Key findings:
1. LOFO revealed 12/21 features were HARMFUL (noise injection)
2. Core signal: CASH_RATIO + EMPLOYEE_COUNT + NET_PROFIT_MARGIN
3. Best new features: FE_GROWTH_QUALITY, FE_DEBT_SERVICE, FE_PROFIT_EFFICIENCY
4. DEBT_RATIO='자본잠식' -> 11.2% default rate (2.98x average)
5. BNPL data: 100% missing for Y=1 companies
""")
