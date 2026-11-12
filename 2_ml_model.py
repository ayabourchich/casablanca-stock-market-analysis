"""
Traitement des actions de la Bourse de Casablanca avec deux modèles ML (pas les indices):

  - Random Forest       → PRIX du lendemain (régression)
  - Logistic Regression → TREND HAUSSE / BAISSE du lendemain (classification)

Produit :
  output/*_predictions.csv        (un fichier par action)
  output/ml_results_summary.csv   (accuracy trend + R² prix de toutes les actions)
  output/feature_importance.csv   (importance des variables Random Forest par action)
  models/random_forest_price.pkl
  models/logistic_regression_trend.pkl
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (mean_absolute_error, mean_squared_error,
                                    r2_score, accuracy_score)


# CHEMINS
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
STOCK_DIR   = os.path.join(BASE_DIR, "cse-data", "stock")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# CONFIGURATION ML
LAG_DAYS    = 5     # jours de lag comme features
TRAIN_YEARS = 4     # entraîner sur les N dernières années seulement
MIN_ROWS    = 100   # ignorer les actions avec trop peu de données

# Features communes (toutes calculées sur le PASSÉ uniquement → pas de leakage)
FEATURES = (
    [f"Close_lag{i}" for i in range(1, LAG_DAYS + 1)] +
    ["Return_lag1", "Return_lag2",
     "MA7", "MA20", "MA50",
     "Volatility", "Volume_MA20"]
)
# CONSTRUCTION DES FEATURES
def build_features(df):
    df = df.copy()
    # Lags de prix (features passées uniquement)
    for lag in range(1, LAG_DAYS + 1):
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    # Rendements passés réels (entre t-2→t-1 et t-3→t-2)
    df["Return_lag1"] = df["Close"].pct_change().shift(1) * 100
    df["Return_lag2"] = df["Close"].pct_change().shift(2) * 100
    #  CIBLES 
    # Prix du lendemain (régression → Random Forest)
    df["Target_Price"] = df["Close"].shift(-1)
    # Trend du lendemain : 1 = HAUSSE, 0 = BAISSE (classification → Logistic)
    df["Target_Trend"]  = (df["Target_Price"] > df["Close"]).astype(int)
    df["Target_Label"]  = df["Target_Trend"].map({1: "HAUSSE ↑", 0: "BAISSE ↓"})
    return df
# ENTRAÎNEMENT D'UNE ACTION
def train_one(symbol, save_models=False):
    filepath = os.path.join(OUTPUT_DIR, f"{symbol}_clean.csv")
    if not os.path.exists(filepath):
        return None, None
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = build_features(df)
    needed = list(set(FEATURES + ["Target_Price", "Target_Trend"]))
    df     = df.dropna(subset=needed)
    # Garder seulement les TRAIN_YEARS dernières années
    cutoff = df["Date"].max() - pd.DateOffset(years=TRAIN_YEARS)
    df     = df[df["Date"] >= cutoff].copy()
    if len(df) < MIN_ROWS:
        print(f"    {symbol:<6} — ignoré ({len(df)} lignes insuffisantes)")
        return None, None
    # Split temporel 80/20 (pas de shuffle → respect de l'ordre chronologique)
    split    = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    # Garde-fou : la classification a besoin des deux classes dans le train
    if df_train["Target_Trend"].nunique() < 2:
        print(f"    {symbol:<6} — ignoré (une seule classe trend dans le train)")
        return None, None
    #  Random Forest → Prix du lendemain 
    # (pas besoin de scaling pour un arbre, on utilise les features brutes)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    rf.fit(df_train[FEATURES], df_train["Target_Price"])
    pred_price   = rf.predict(df_test[FEATURES])
    actual_price = df_test["Target_Price"].values
    mae  = round(mean_absolute_error(actual_price, pred_price), 2)
    rmse = round(np.sqrt(mean_squared_error(actual_price, pred_price)), 2)
    r2   = round(r2_score(actual_price, pred_price), 4)
    importance = pd.Series(
        rf.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    # Logistic Regression → Trend HAUSSE/BAISSE
    # (la régression logistique a besoin de features scalées)
    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(df_train[FEATURES])
    Xte     = scaler.transform(df_test[FEATURES])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, df_train["Target_Trend"])
    pred_trend  = clf.predict(Xte)
    actual_trend = df_test["Target_Trend"].values
    trend_acc   = round(accuracy_score(actual_trend, pred_trend) * 100, 2)
    pred_label  = pd.Series(pred_trend).map({1: "HAUSSE ↑", 0: "BAISSE ↓"}).values
    # Sauvegarder les modèles pkl (une seule fois, comme référence)
    if save_models:
        joblib.dump({"model": rf, "features": FEATURES},
                    os.path.join(MODELS_DIR, "random_forest_price.pkl"))
        joblib.dump({"model": clf, "scaler": scaler, "features": FEATURES},
                    os.path.join(MODELS_DIR, "logistic_regression_trend.pkl"))
    # Export prédictions
    df_out = df_test.copy()
    df_out["Pred_Price_RF"]    = pred_price
    df_out["Pred_Price_Error"] = abs(pred_price - actual_price)
    df_out["Pred_Trend"]       = pred_trend
    df_out["Pred_Trend_Label"] = pred_label
    cols = ["Date", "Symbol", "Close", "Volume",
            "Target_Price", "Pred_Price_RF", "Pred_Price_Error",
            "Target_Trend", "Target_Label",
            "Pred_Trend", "Pred_Trend_Label"]
    df_out[[c for c in cols if c in df_out.columns]].to_csv(
        os.path.join(OUTPUT_DIR, f"{symbol}_predictions.csv"), index=False
    )

    result = {
        "symbol"        : symbol,
        "trend_accuracy": trend_acc,
        "price_mae"     : mae,
        "price_rmse"    : rmse,
        "price_r2"      : r2,
        "n_train"       : len(df_train),
        "n_test"        : len(df_test),
        "top_feature"   : importance.index[0],
        "top_importance": round(importance.iloc[0] * 100, 1)
    }
    return result, importance

# MAIN — traiter les 77 actions
print("-" * 62)
print("MODÈLE ML — Bourse de Casablanca (77 actions)")
print("Random Forest → Prix  |  Logistic Regression → Trend")
print("-" * 62)
stock_files = sorted(glob.glob(os.path.join(STOCK_DIR, "*.csv")))
all_symbols = [os.path.basename(f).replace(".csv", "") for f in stock_files]
print(f"\n {len(all_symbols)} actions détectées\n")
results         = []
all_importances = []
first_saved     = False

for symbol in all_symbols:
    save = not first_saved
    res, imp = train_one(symbol, save_models=save)
    if res:
        results.append(res)
        if not first_saved:
            first_saved = True
        if imp is not None:
            imp_df = imp.reset_index()
            imp_df.columns = ["Feature", "Importance"]
            imp_df["Symbol"] = symbol
            all_importances.append(imp_df)
        if res["price_r2"] >= 0.5:
            status = "🟢"
        elif res["price_r2"] >= 0:
             status = "🟡"
        else:
             status = "🔴"
        print(f"  {status} {symbol:<6}  "
              f"Trend: {res['trend_accuracy']:>5}%  |  "
              f"MAE: {res['price_mae']:>8.2f}  |  "
              f"R²: {res['price_r2']:>6.4f}")

# Exports
if not results:
    print("\n Aucune action traitée — lancer 1_nettoyage.py d'abord ?")
    raise SystemExit(1)

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "ml_results_summary.csv"), index=False)

if all_importances:
    pd.concat(all_importances, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
    )

# RÉCAPITULATIF
good   = df_results[df_results["price_r2"] >= 0.5]
medium = df_results[(df_results["price_r2"] >= 0) & (df_results["price_r2"] < 0.5)]
bad    = df_results[df_results["price_r2"] < 0]

print("\n" + "-" * 62)
print("RÉCAPITULATIF")
print("-" * 62)
print(f"  Actions traitées       : {len(results)}/{len(all_symbols)}")
print(f"  🟢 R² ≥ 0.5            : {len(good)} actions — bonne prédiction prix")
print(f"  🟡 0 ≤ R² < 0.5        : {len(medium)} actions — prédiction correcte")
print(f"  🔴 R² < 0              : {len(bad)} actions — haute volatilité")
if len(bad) > 0:
    print(f"     ({', '.join(bad['symbol'].tolist())})")
print(f"\n  Trend accuracy moy     : {df_results['trend_accuracy'].mean():.1f}%")
print(f"  R² moyen               : {df_results['price_r2'].mean():.4f}")
print(f"\n   Modèles sauvegardés dans models/")
print(f"   Résultats dans output/")
print("\n Modèle ML terminé avec succès !")
