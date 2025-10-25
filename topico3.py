# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV

# =========================
# 0) Carregamento de dados
# =========================
CSV_PATH = Path("clientes_churn.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Arquivo {CSV_PATH} não encontrado. Coloque o CSV na mesma pasta."
    )

df = pd.read_csv(CSV_PATH)

# Alvo e features
TARGET = "churn"
if TARGET not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET}' não encontrada no CSV.")

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# Identificar tipos de colunas
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================================
# 1) Pipeline (pré-processamento + classificador)
# =============================================
# Observação: StandardScaler(with_mean=False) lida melhor com esparsidade pós OneHot
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# Escolha do modelo: você pode alternar entre RF e LogReg.
# RF costuma funcionar bem em tabular e tem predict_proba estável.
clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced_subsample",
)

# Se quiser LogReg como baseline, descomente:
# clf = LogisticRegression(max_iter=500)

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf)
])

# Treinar
pipe.fit(X_train, y_train)

# Avaliação rápida
proba_test = pipe.predict_proba(X_test)[:, 1]
pred_test  = (proba_test >= 0.5).astype(int)
print("== Avaliação (pipeline base) ==")
print(f"AUC: {roc_auc_score(y_test, proba_test):.4f}")
print(f"F1 @0.5: {f1_score(y_test, pred_test):.4f}")
print(classification_report(y_test, pred_test))

# ==================================================
# 2) Explicabilidade — Permutation Importance (PI)
# ==================================================
def get_feature_names_after_column_transformer(preprocessor):
    """
    Retorna uma lista de nomes de features após o ColumnTransformer.
    Para numéricos (StandardScaler), mantém o nome original.
    Para categóricos (OneHotEncoder), expande usando get_feature_names_out.
    """
    feature_names = []
    # Loop nos "transformers_" já fitados
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if isinstance(transformer, Pipeline):
            # Caso exista um Pipeline dentro do transformer
            last_step = transformer.steps[-1][1]
        else:
            last_step = transformer

        if hasattr(last_step, "get_feature_names_out"):
            try:
                names = last_step.get_feature_names_out(cols)
            except TypeError:
                names = last_step.get_feature_names_out()
            feature_names.extend(list(names))
        else:
            # Sem get_feature_names_out (ex.: StandardScaler) -> usa nomes originais
            feature_names.extend(list(cols))
    return feature_names

# Precisamos do ColumnTransformer já FITADO
pre_fitted = pipe.named_steps["pre"]
feat_names = get_feature_names_after_column_transformer(pre_fitted)

# PI no conjunto de teste (mantém robustez de generalização)
print("\n== Permutation Importance (PI) ==")
pi = permutation_importance(
    estimator=pipe,
    X=X_test,
    y=y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42
)

# Ajuste: garantir que o número de nomes de features corresponda ao número de importâncias
n_feats = len(pi.importances_mean)
if len(feat_names) != n_feats:
    print(f"[AVISO] Número de nomes de features ({len(feat_names)}) diferente do número de importâncias ({n_feats}). Ajustando...")
    feat_names = feat_names[:n_feats]

# Montar DataFrame com importâncias
pi_df = pd.DataFrame({
    "feature": feat_names,
    "importance_mean": pi.importances_mean,
    "importance_std": pi.importances_std
}).sort_values("importance_mean", ascending=False)

print("\nTop 15 features por importância permutada (AUC):")
print(pi_df.head(15).to_string(index=False))

# Salvar para eventual uso em relatório
pi_df.to_csv("permutation_importance.csv", index=False, encoding="utf-8")
print("Arquivo salvo: permutation_importance.csv")

# ======================================================
# 3) Fairness básica — métricas por subgrupo (ex.: plano)
# ======================================================


GROUP_COL = None
# Escolher automaticamente um grupo plausível se existir:
for c in ["plano", "Plano", "segmento", "Segmento"]:
    if c in X_test.columns:
        GROUP_COL = c
        break

if GROUP_COL is None and len(cat_cols) > 0:
    # Como fallback, usar a primeira coluna categórica encontrada
    GROUP_COL = cat_cols[0]

if GROUP_COL is not None:
    print(f"\n== Métricas por subgrupo (fairness) — grupo: {GROUP_COL} ==")
    test_with_group = X_test.copy()
    test_with_group["_y"] = y_test.values
    test_with_group["_proba"] = proba_test
    test_with_group["_pred"] = pred_test

    # Funções auxiliares
    def rates(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall sensibilidade
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return dict(TPR=tpr, FPR=fpr, Precision=ppv, NPV=npv)

    summary_rows = []
    for grp, part in test_with_group.groupby(GROUP_COL):
        metrics = rates(part["_y"], part["_pred"])
        auc_g = roc_auc_score(part["_y"], part["_proba"]) if part["_y"].nunique() == 2 else np.nan
        row = {"grupo": grp, "n": len(part), "AUC": auc_g, **metrics}
        summary_rows.append(row)

    fairness_df = pd.DataFrame(summary_rows).sort_values("grupo").reset_index(drop=True)
    print(fairness_df.to_string(index=False))

    # Disparidades simples (máx - mín) como sinal de alerta
    def disparity(series):
        if series.empty or series.isna().all():
            return np.nan
        return series.max() - series.min()

    print("\nDisparidades (max - min) entre grupos:")
    for col in ["AUC", "TPR", "FPR", "Precision", "NPV"]:
        print(f"  {col}: {disparity(fairness_df[col]):.4f}")

    # Exportar para análise externa
    fairness_df.to_csv("fairness_grupos.csv", index=False, encoding="utf-8")
    print("Arquivo salvo: fairness_grupos.csv")
else:
    print("\n[AVISO] Nenhuma coluna categórica encontrada para análise de subgrupos (fairness).")

# =====================================
# 4) Calibração de probabilidades (Platt/Isotonic)
# =====================================

print("\n== Calibração (Isotonic) ==")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cal = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=cv)
cal.fit(X_train, y_train)

proba_cal = cal.predict_proba(X_test)[:, 1]
pred_cal = (proba_cal >= 0.5).astype(int)

print(f"AUC (calibrado): {roc_auc_score(y_test, proba_cal):.4f}")
print(f"F1 (calibrado) @0.5: {f1_score(y_test, pred_cal):.4f}")

# (Opcional) Fairness após calibração — repetir tabela por grupo
if GROUP_COL is not None:
    test_with_group["_proba_cal"] = proba_cal
    test_with_group["_pred_cal"] = pred_cal
    rows_cal = []
    for grp, part in test_with_group.groupby(GROUP_COL):
        metrics = rates(part["_y"], part["_pred_cal"])
        auc_g = roc_auc_score(part["_y"], part["_proba_cal"]) if part["_y"].nunique() == 2 else np.nan
        rows_cal.append({"grupo": grp, "n": len(part), "AUC_cal": auc_g, **metrics})
    fairness_cal_df = pd.DataFrame(rows_cal).sort_values("grupo").reset_index(drop=True)
    print("\nMétricas por grupo (calibrado):")
    print(fairness_cal_df.to_string(index=False))
    fairness_cal_df.to_csv("fairness_grupos_calibrado.csv", index=False, encoding="utf-8")
    print("Arquivo salvo: fairness_grupos_calibrado.csv")

print("\nConcluído. Arquivos gerados:")
print(" - permutation_importance.csv")
if GROUP_COL is not None:
    print(" - fairness_grupos.csv")
    print(" - fairness_grupos_calibrado.csv (após calibração)")