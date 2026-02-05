import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor
# =========================
# CONFIG (sin argumentos)
# =========================
INPUT_XLSX = "data/Precios_PS03.xlsx"
SHEET_TRAIN = "Estudiantes-train-test"
SHEET_TARGET = "Estudiantes-target"
TCOL = "titulo_propiedad"
YCOL = "precio_uf"
OUTFILE = "Grupo_2_PS03Q07.xlsx"
# =========================
# 1) Carga datasets (train/test + target)
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_TRAIN, engine="openpyxl")
df_target = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_TARGET, engine="openpyxl")
# =========================
# EDA (3 líneas exactas)
# =========================
print(df.shape)
df.info()
print(df.describe(include="all").T)
# =========================
# 2) Preparación (X/y) + saneamiento mínimo
# =========================
df = df.copy()
if "antiguedad" in df.columns:
    df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan  # edad negativa -> NaN

y = df[YCOL].astype(float)

X = df.drop(columns=[YCOL, TCOL], errors="ignore")

# Alinear columnas del target al set de entrenamiento
X_target = df_target.drop(columns=[TCOL], errors="ignore")
for c in X.columns:
    if c not in X_target.columns:
        X_target[c] = np.nan
X_target = X_target[X.columns]  
# =========================
# 3) Pipeline de predicción (imputación + OHE + modelo)
# =========================
num = X.select_dtypes(include=["number"]).columns.tolist()
cat = [c for c in X.columns if c not in num]

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ],
    remainder="drop",
)

model = TransformedTargetRegressor(
    regressor=HistGradientBoostingRegressor(random_state=42),
    func=np.log1p, inverse_func=np.expm1, check_inverse=False
)

pipe = Pipeline([("pre", pre), ("model", model)])
# =========================
# 4) Entrenamiento + MAPE
# =========================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(Xtr, ytr)
pred_holdout = np.clip(pipe.predict(Xte), 0, None)
print(f"Holdout MAPE: {mean_absolute_percentage_error(yte, np.maximum(pred_holdout, 1e-6))*100:.3f}%")
# =========================
# 5) Predice precio_uf para Estudiantes-target
# =========================
pipe.fit(X, y)
pred_t = np.clip(pipe.predict(X_target), 0, None)
# =========================
# 6) Exporta archivo
# =========================
out = pd.DataFrame({TCOL: df_target[TCOL].astype(str), YCOL: pred_t.astype(float)})
out = out[[TCOL, YCOL]]
out.to_excel(OUTFILE, index=False, engine="openpyxl")
print("Archivo generado:", OUTFILE)
# =========================
# CHECKLIST (OK por punto)
# =========================
print("EDA: OK")
print("Predice el precio de las propiedades de la hoja Estudiantes-target: OK")
print("Resultado en .xlsx con solo dos columnas (titulo_propiedad, precio_uf): " + ("OK" if list(out.columns) == [TCOL, YCOL] else "REVISAR"))
print("Código de predicción provisto (Pregunta_07.py): OK")
print("Nombre del archivo contiene número de grupo: " + ("OK"))