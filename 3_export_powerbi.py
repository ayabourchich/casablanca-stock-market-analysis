"""

Génèration de  8 fichiers CSV pour powerbi_export/ à partir des données nettoyées et des résultats ML :
  pbi_masi.csv                  ← évolution MASI
  pbi_all_indices.csv           ← tous les 32 indices combinés
  pbi_stocks_comparison.csv     ← prix + volume toutes les 77 actions
  pbi_predictions.csv           ← prédictions prix (RF) + trend (Logistic)
  pbi_annual_performance.csv    ← performance % par action par année
  pbi_ml_summary.csv            ← accuracy + R² par action (KPI cards)
  pbi_feature_importance.csv    ← importance des variables (Random Forest)
  pbi_stocks_info.csv           ← noms complets des actions
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings("ignore")

# CHEMINS
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
POWERBI_DIR = os.path.join(BASE_DIR, "powerbi_export")
DATA_DIR    = os.path.join(BASE_DIR, "cse-data")
os.makedirs(POWERBI_DIR, exist_ok=True)

print("-" * 56)
print("fichiers CSV pour Power BI ")
print("-" * 56)


# UTILITAIRES
def load_clean(symbol):
    """Charge le fichier nettoyé d'un symbole (insensible à la casse)."""
    fp = os.path.join(OUTPUT_DIR, f"{symbol}_clean.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, parse_dates=["Date"])
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*_clean.csv")):
        base = os.path.basename(f).replace("_clean.csv", "")
        if base.lower() == symbol.lower():
            return pd.read_csv(f, parse_dates=["Date"])
    return None

def get_all_symbols(subfolder):
    """Retourne tous les symboles d'un sous-dossier cse-data/."""
    files = glob.glob(os.path.join(DATA_DIR, subfolder, "*.csv"))
    return sorted([os.path.basename(f).replace(".csv", "") for f in files])

def safe_cols(df, cols):
    """Ne garde que les colonnes existantes (évite les KeyError)."""
    return df[[c for c in cols if c in df.columns]]

STOCK_SYMBOLS = get_all_symbols("stock")
INDEX_SYMBOLS = get_all_symbols("index")
print(f"\n {len(STOCK_SYMBOLS)} actions  |  {len(INDEX_SYMBOLS)} indices\n")

# TABLE 1 : Évolution du MASI
print(" Table 1 : MASI...")
df_masi = load_clean("MASI")
if df_masi is not None:
    cols = ["Date", "Close", "MA20", "MA50", "Daily_Return", "Volatility", "Volume"]
    safe_cols(df_masi, cols).rename(
        columns={"Close": "MASI_Close", "MA20": "MASI_MA20",
                 "MA50": "MASI_MA50", "Daily_Return": "MASI_Return"}
    ).to_csv(os.path.join(POWERBI_DIR, "pbi_masi.csv"), index=False)
    print(f"   pbi_masi.csv — {len(df_masi)} lignes")
else:
    print("    MASI_clean.csv non trouvé — lancer 1_nettoyage.py d'abord")

# TABLE 2 : Tous les indices combinés

print("\n Table 2 : Tous les indices...")
idx_frames = []
for sym in INDEX_SYMBOLS:
    df = load_clean(sym)
    if df is not None:
        idx_frames.append(safe_cols(
            df, ["Date", "Symbol", "Close", "MA20", "Daily_Return", "Volatility"]))

if idx_frames:
    df_indices = pd.concat(idx_frames, ignore_index=True)
    df_indices.to_csv(os.path.join(POWERBI_DIR, "pbi_all_indices.csv"), index=False)
    print(f"   pbi_all_indices.csv — {len(df_indices):,} lignes "
          f"({len(idx_frames)} indices)")
else:
    print("    Aucun indice trouvé")

# TABLE 3 : Comparaison des 77 actions
print("\n Table 3 : Comparaison des actions...")
stock_frames = []
for sym in STOCK_SYMBOLS:
    df = load_clean(sym)
    if df is not None:
        stock_frames.append(safe_cols(
            df, ["Date", "Symbol", "Close", "Volume", "Daily_Return",
                 "MA7", "MA20", "Volatility"]))
if stock_frames:
    df_compare = pd.concat(stock_frames, ignore_index=True)
    df_compare.to_csv(os.path.join(POWERBI_DIR, "pbi_stocks_comparison.csv"), index=False)
    print(f"   pbi_stocks_comparison.csv — {len(df_compare):,} lignes "
          f"({len(stock_frames)} actions)")
else:
    print("  Aucune action trouvée")

# TABLE 4 : Prédictions (toutes les actions)
print("\n Table 4 : Prédictions...")
pred_frames = []
for sym in STOCK_SYMBOLS:
    fp = os.path.join(OUTPUT_DIR, f"{sym}_predictions.csv")
    if os.path.exists(fp):
        pred_frames.append(pd.read_csv(fp, parse_dates=["Date"]))
if pred_frames:
    df_pred = pd.concat(pred_frames, ignore_index=True)
    df_pred.to_csv(os.path.join(POWERBI_DIR, "pbi_predictions.csv"), index=False)
    print(f"   pbi_predictions.csv — {len(df_pred):,} lignes "
          f"({len(pred_frames)} actions)")
else:
    print("    Aucune prédiction — lancer 2_ml_model.py d'abord")

# TABLE 5 : Performance annuelle
print("\n Table 5 : Performance annuelle...")
perf_rows = []
for sym in STOCK_SYMBOLS:
    df = load_clean(sym)
    if df is None or df.empty:
        continue
    df["Year"] = df["Date"].dt.year
    for year, grp in df.groupby("Year"):
        open_p  = grp["Close"].iloc[0]
        close_p = grp["Close"].iloc[-1]
        perf    = ((close_p - open_p) / open_p) * 100 if open_p != 0 else 0
        perf_rows.append({
            "Symbol"          : sym,
            "Year"            : year,
            "Open_Year"       : round(open_p, 2),
            "Close_Year"      : round(close_p, 2),
            "Performance_Pct" : round(perf, 2),
            "Avg_Volume"      : round(grp["Volume"].mean(), 0),
            "Max_Price"       : round(grp["High"].max(), 2) if "High" in grp else round(grp["Close"].max(), 2),
            "Min_Price"       : round(grp["Low"].min(), 2) if "Low" in grp else round(grp["Close"].min(), 2),
        })

if perf_rows:
    df_perf = pd.DataFrame(perf_rows)
    df_perf.to_csv(os.path.join(POWERBI_DIR, "pbi_annual_performance.csv"), index=False)
    print(f"   pbi_annual_performance.csv — {len(df_perf):,} lignes")
else:
    print("    Aucune performance calculée")

# TABLE 6 : Résumé ML
print("\n Table 6 : Résumé ML...")
ml_fp = os.path.join(OUTPUT_DIR, "ml_results_summary.csv")
if os.path.exists(ml_fp):
    df_ml = pd.read_csv(ml_fp)
    df_ml.to_csv(os.path.join(POWERBI_DIR, "pbi_ml_summary.csv"), index=False)
    print(f"   pbi_ml_summary.csv — {len(df_ml)} actions")
else:
    print("    ml_results_summary.csv non trouvé — lancer 2_ml_model.py d'abord")

# TABLE 7 : Feature importance
print("\n Table 7 : Feature importance...")
fi_fp = os.path.join(OUTPUT_DIR, "feature_importance.csv")
if os.path.exists(fi_fp):
    pd.read_csv(fi_fp).to_csv(
        os.path.join(POWERBI_DIR, "pbi_feature_importance.csv"), index=False
    )
    print(f"   pbi_feature_importance.csv")
else:
    print("    feature_importance.csv non trouvé — lancer 2_ml_model.py d'abord")

# TABLE 8 : Infos actions
print("\n Table 8 : Infos actions...")
info_fp = os.path.join(DATA_DIR, "info.csv")
if os.path.exists(info_fp):
    pd.read_csv(info_fp).to_csv(
        os.path.join(POWERBI_DIR, "pbi_stocks_info.csv"), index=False
    )
    print(f"   pbi_stocks_info.csv")
else:
    print("    cse-data/info.csv non trouvé")

# RÉSUMÉ
files = [f for f in os.listdir(POWERBI_DIR) if f.endswith(".csv")]
print("\n" + "-" * 56)
print(f"FICHIERS EXPORTÉS ({len(files)}) → powerbi_export/")
print("-" * 56)
for f in sorted(files):
    size = os.path.getsize(os.path.join(POWERBI_DIR, f))
    print(f"   {f:<40} {size//1024:>4} KB")

print("Export Power BI terminé avec succès !")
