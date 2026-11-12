"""
data processing script for Casablanca Stock Exchange (CSE) data:
  - 77 actions  → cse-data/stock/*.csv
  - 32 indices  → cse-data/index/*.csv

Produit :
  output/*_clean.csv          (un fichier par action/indice)
  output/all_stocks_clean.csv (toutes les actions combinées)
  output/all_indices_clean.csv (tous les indices combinés)
  output/masi_clean.csv       (MASI séparé pour référence rapide)
"""

import pandas as pd
import numpy as np
import os
import glob

# ─────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "cse-data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
STOCK_DIR   = os.path.join(DATA_DIR, "stock")
INDEX_DIR   = os.path.join(DATA_DIR, "index")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# FONCTION : Charger et nettoyer un fichier CSV
# ─────────────────────────────────────────────
def load_and_clean(filepath, symbol):
    df = pd.read_csv(filepath)
    # Renommer la colonne date
    df.rename(columns={"Time": "Date"}, inplace=True)
    # Convertir en datetime et trier
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Gérer les valeurs manquantes
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    # Ajouter symbole
    df["Symbol"] = symbol
    # Garder les colonnes utiles
    df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
    return df


# ─────────────────────────────────────────────
# FONCTION : Ajouter les indicateurs techniques
# ─────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    # Moyennes mobiles
    df["MA7"]  = df["Close"].rolling(window=7,  min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    # Volatilité (écart-type 20j)
    df["Volatility"]   = df["Close"].rolling(window=20, min_periods=2).std()
    # Rendement journalier %
    df["Daily_Return"] = df["Close"].pct_change() * 100
    # Volume moyen 20j
    df["Volume_MA20"]  = df["Volume"].rolling(window=20, min_periods=1).mean()
    return df

# ─────────────────────────────────────────────
# TRAITEMENT DES ACTIONS
# ─────────────────────────────────────────────
print("=" * 56)
print("NETTOYAGE DES DONNÉES — Bourse de Casablanca")
print("=" * 56)
stock_files = sorted(glob.glob(os.path.join(STOCK_DIR, "*.csv")))
index_files = sorted(glob.glob(os.path.join(INDEX_DIR, "*.csv")))
print(f"\nfichiers Trouvés : {len(stock_files)} actions  |  {len(index_files)} indices\n")

# ── Actions ──────────────────────────────────
print(" Traitement des actions->stocks...")
all_stocks = []
errors     = []

for filepath in stock_files:
    symbol = os.path.basename(filepath).replace(".csv", "")
    try:
        df = load_and_clean(filepath, symbol)
        df = add_indicators(df)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{symbol}_clean.csv"), index=False)
        all_stocks.append(df)
        print(f"  done: {symbol:<6} — {len(df):>4} lignes  "
              f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    except Exception as e:
        errors.append(symbol)
        print(f"   {symbol:<6} — ERREUR : {e}")

# Fichier combiné toutes actions
if all_stocks:
    df_all = pd.concat(all_stocks, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_clean.csv"), index=False)
    print(f"\n   all_stocks_clean.csv — {len(df_all):,} lignes au total")


# ── Indices ───────────────────────────────────
print("\n Traitement des indices...")
all_indices = []

for filepath in index_files:
    symbol = os.path.basename(filepath).replace(".csv", "")
    try:
        df = load_and_clean(filepath, symbol)
        df = add_indicators(df)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{symbol}_clean.csv"), index=False)
        all_indices.append(df)
        print(f"  done: {symbol:<8} — {len(df):>4} lignes  "
              f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    except Exception as e:
        errors.append(symbol)
        print(f"   {symbol:<8} — ERREUR : {e}")

# Fichier combiné tous indices
if all_indices:
    df_idx = pd.concat(all_indices, ignore_index=True)
    df_idx.to_csv(os.path.join(OUTPUT_DIR, "all_indices_clean.csv"), index=False)
    print(f"\n   all_indices_clean.csv — {len(df_idx):,} lignes au total")

# MASI séparé pour accès rapide
masi_path = os.path.join(OUTPUT_DIR, "MASI_clean.csv")
masi_dest = os.path.join(OUTPUT_DIR, "masi_clean.csv")
if os.path.exists(masi_path) and os.path.abspath(masi_path).lower() != os.path.abspath(masi_dest).lower():
    import shutil
    shutil.copy(masi_path, masi_dest)
    print(f"   masi_clean.csv — copie de MASI_clean.csv")
else:
    print(f"   masi_clean.csv — identique à MASI_clean.csv (Windows)")


# ─────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 56)
print("RÉSUMÉ")
print("=" * 56)
print(f"  Actions traitées   : {len(all_stocks)}/{len(stock_files)}")
print(f"  Indices traités    : {len(all_indices)}/{len(index_files)}")
if errors:
    print(f"   Erreurs         : {', '.join(errors)}")
print(f"  Fichiers dans      : output/")
print("\n data processing completed successfully! \n")
