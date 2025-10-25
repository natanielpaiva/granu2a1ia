# requirements: scikit-learn, pandas, numpy
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Exemplo: dataset tabular (pode usar o clientes_churn.csv da U1A4)
df = pd.read_csv("clientes_churn.csv")
y = df["churn"].astype(int)
X = df.drop(columns=["churn"])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(with_mean=False), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=500))
])

# Split com estratificação e holdout de teste
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)

# Validação cruzada estratificada no treino
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
print("CV AUC (mean±std):", scores.mean(), "±", scores.std())