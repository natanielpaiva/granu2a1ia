# requirements: tensorflow, scikit-learn, pandas, numpy
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregamento do dataset
df = pd.read_csv("clientes_churn.csv")
# Identificação das colunas numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col != "churn"]

# Ex.: usar apenas colunas numéricas para o MLP (ou crie dummies para categorias)
X_num = df[num_cols].copy()
scaler = StandardScaler(with_mean=False)
Xn = scaler.fit_transform(X_num.values)
y = df["churn"].astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(Xn, y, test_size=0.2,
                                          stratify=y, random_state=42)

inputs = tf.keras.Input(shape=(X_tr.shape[1],))
x = tf.keras.layers.Dense(64, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])

es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                      patience=10, restore_best_weights=True)
hist = model.fit(X_tr, y_tr, epochs=100, batch_size=64,
                 validation_split=0.2, callbacks=[es], verbose=0)

# Avaliação
proba = model.predict(X_te).ravel()
from sklearn.metrics import roc_auc_score
print("Test AUC (MLP):", roc_auc_score(y_te, proba))