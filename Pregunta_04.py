import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, cast, Optional, Dict, List, Tuple
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented", category=pd.errors.PerformanceWarning)

def run() -> None:
    # =========================
    # Lectura de archivos
    # =========================
    sales = pd.read_csv("data/sales_train_evaluation.csv")
    cal = pd.read_csv(
        "data/calendar.csv",
        usecols=["d", "date", "wm_yr_wk", "wday", "snap_CA", "event_name_1", "event_name_2"],
    )
    prices = pd.read_csv(
        "data/sell_prices.csv",
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    # =========================
    # a) Preparación de datos
    # =========================
    print("\n=== a) Preparación de datos ===")

    stores = ["CA_1", "CA_2", "CA_3", "CA_4"]
    dept = "FOODS_3"
    dcols = [f"d_{i}" for i in range(1, 1914)]
    sales = sales[(sales.dept_id == dept) & (sales.store_id.isin(stores))]

    meta = sales[["item_id", "store_id", "dept_id"]].drop_duplicates()
    p = (
        prices.merge(meta, on=["item_id", "store_id"], how="inner")
        .groupby(["store_id", "dept_id", "wm_yr_wk"], as_index=False)["sell_price"]
        .mean()
    )

    wide = sales.groupby(["store_id", "dept_id"], as_index=False)[dcols].sum()
    wide = wide.copy()

    df = wide.melt(["store_id", "dept_id"], var_name="d", value_name="y")
    df = df.merge(cal, on="d", how="left").merge(p, on=["store_id", "dept_id", "wm_yr_wk"], how="left")

    df["unique_id"] = df["store_id"].astype(str).str.cat(df["dept_id"].astype(str), sep="_")
    df["ds"] = pd.to_datetime(df["date"])
    df["is_event"] = ((df["event_name_1"].notna()) | (df["event_name_2"].notna())).astype(int)
    df["day_of_week"] = df["wday"].astype(int)

    # M5: 1=sáb, 2=dom ... → weekend = {1,2}
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(int)

    df["week_of_year"] = df["ds"].apply(lambda x: int(x.isocalendar()[1]))
    df["d_num"] = df["d"].str.replace("d_", "", regex=False).astype(int)

    df = (
        df[
            [
                "unique_id",
                "ds",
                "y",
                "is_event",
                "snap_CA",
                "sell_price",
                "day_of_week",
                "is_weekend",
                "week_of_year",
                "d_num",
            ]
        ]
        .sort_values(["unique_id", "ds"])
        .dropna()
    )

    print("\nEntregable (a): Código de preparación y .head() del DataFrame resultante.")
    print(df.head())
    # =========================
    # EDA
    # =========================
    print(df.shape)
    df.info()
    print(df.describe(include="all").T)
    # =========================
    # b) Configuración del pipeline MLForecast
    # =========================
    print("\n=== b) Configuración del pipeline MLForecast ===")

    tr = df[df.d_num <= 1900].drop(columns="d_num")
    te = df[(df.d_num >= 1901) & (df.d_num <= 1913)].drop(columns="d_num")
    Xte = te.drop(columns="y")

    models = [
        LGBMRegressor(random_state=0, verbosity=-1),
        XGBRegressor(
            random_state=0,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
        ),
    ]
    lag_t = cast(Any, {1: [RollingMean(7), RollingMean(14), RollingStd(7)]})

    fcst = MLForecast(
        models=models,
        freq="D",
        lags=[1, 7, 14, 28],
        lag_transforms=lag_t,
        date_features=["day", "dayofyear", "week"],
    )

    fcst.fit(tr, static_features=[])
    print("\nEntregable (b): Código completo del pipeline y confirmación de entrenamiento exitoso.")
    print("Entrenamiento MLForecast: OK")
    # =========================
    # c) Predicción y evaluación
    # =========================
    print("\n=== c) Predicción y evaluación ===")

    pred = fcst.predict(h=13, X_df=Xte).merge(te, on=["unique_id", "ds"], how="left")
    model_cols = [m.__class__.__name__ for m in models]
    model_cols = [c for c in model_cols if c in pred.columns]

    def mape(y, yp):
        y = np.asarray(y)
        yp = np.asarray(yp)
        m = y != 0
        return np.nan if m.sum() == 0 else np.mean(np.abs((y[m] - yp[m]) / y[m])) * 100

    rows = []
    for m in model_cols:
        y = pred["y"].values
        yp = pred[m].values
        rows.append([m, mean_absolute_error(y, yp), root_mean_squared_error(y, yp), mape(y, yp)])

    metrics = pd.DataFrame(rows, columns=["model", "MAE", "RMSE", "MAPE_%"]).sort_values("RMSE").reset_index(drop=True)

    print("\nEntregable (c): Tabla comparativa de métricas y gráfico de predicciones.")
    print(metrics.round({"MAE": 3, "RMSE": 3, "MAPE_%": 3}).to_string(index=False))

    uid = df["unique_id"].iloc[0]
    hist = tr[tr.unique_id == uid].tail(60)[["ds", "y"]]
    real = te[te.unique_id == uid][["ds", "y"]]
    pp = pred[pred.unique_id == uid][["ds"] + model_cols]

    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"forecast_{uid}.png"

    plt.figure(figsize=(10, 4))
    plt.plot(hist.ds, hist.y, label="train (last 60)")
    plt.plot(real.ds, real.y, label="test real")
    for m in model_cols:
        plt.plot(pp.ds, pp[m], label=f"pred {m}")
    plt.title(uid)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.legend()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print("\nGráfico guardado en:", outfile.as_posix())
    plt.show()
    # =========================
    # d) Feature importance
    # =========================
    print("\n=== d) Feature importance (para ambos modelos) ===")
    best = str(metrics.iloc[0]["model"])
    

    fitted = getattr(fcst, "models_", None)
    if fitted is None:
        fitted = getattr(fcst, "fitted_models_", None)
    if fitted is None:
        fitted = getattr(fcst, "fitted_models", None)

    def _iter_fitted_models() -> List[Tuple[str, Any]]:
        if isinstance(fitted, dict):
            out = []
            for _, mdl in fitted.items():
                out.append((mdl.__class__.__name__, mdl))
            return out
        if isinstance(fitted, (list, tuple)):
            return [(mdl.__class__.__name__, mdl) for mdl in fitted]
        return [(mdl.__class__.__name__, mdl) for mdl in models]

    def _show_importance(name: str, mdl: Any, topn: int = 15) -> None:
        # 1) LightGBM
        fi = getattr(mdl, "feature_importances_", None)
        if fi is not None:
            fi = np.asarray(fi).reshape(-1)

            feat_names = getattr(mdl, "feature_name_", None)
            if feat_names is None or len(feat_names) != len(fi):
                feat_names = [f"f{i}" for i in range(len(fi))]

            s = pd.Series(fi, index=list(feat_names)).sort_values(ascending=False).head(topn)
            print(f"\nTop {topn} feature importance [{name}] (feature_importances_):")
            print(s.to_string())
            return

        # 2) XGB
        if hasattr(mdl, "get_booster"):
            try:
                b = mdl.get_booster()
                score = b.get_score(importance_type="gain")  
                if score:
                    s = pd.Series(score).sort_values(ascending=False).head(topn)
                    print(f"\nTop {topn} feature importance [{name}] (XGB gain):")
                    print(s.to_string())
                    return
            except Exception:
                pass

        print(f"\n{name}: feature importance no disponible.")

    for name, mdl in _iter_fitted_models():
        if name in ("LGBMRegressor", "XGBRegressor"):
            _show_importance(name, mdl, topn=15)
    print("\nModelo recomendado:", best)
    print("\nEntregable (d): Análisis escrito va en el PDF.")

if __name__ == "__main__":
    run()
