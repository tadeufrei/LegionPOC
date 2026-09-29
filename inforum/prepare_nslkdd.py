"""
prepare_nslkdd.py
=================
Downloads, preprocesses, and saves the NSL-KDD dataset for the
LegionITS membership inference experiment.

NSL-KDD is a standard benchmark for intrusion detection FL papers.
Unlike the CAN bus dataset, its 122 features and class imbalance
create conditions where models overfit on small subsets, making
membership inference empirically demonstrable.

Usage:
  python prepare_nslkdd.py

Output:
  dataset_nslkdd/train.csv   — preprocessed training set
  dataset_nslkdd/test.csv    — preprocessed test set

Download (if script cannot reach GitHub automatically):
  1. Go to https://www.unb.ca/cic/datasets/nsl.html
     OR https://github.com/defcom17/NSL_KDD
  2. Download KDDTrain+.txt and KDDTest+.txt
  3. Place both files in the same directory as this script
  4. Run: python prepare_nslkdd.py --local
"""

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Column definitions ─────────────────────────────────────────────────
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level',
]
CAT_COLS = ['protocol_type', 'service', 'flag']
NUM_COLS = [c for c in COLUMNS
            if c not in CAT_COLS + ['label', 'difficulty_level']]

# GitHub mirror (primary source)
TRAIN_URL = ("https://raw.githubusercontent.com/defcom17/"
             "NSL_KDD/master/KDDTrain%2B.txt")
TEST_URL  = ("https://raw.githubusercontent.com/defcom17/"
             "NSL_KDD/master/KDDTest%2B.txt")

OUTPUT_DIR = "dataset_nslkdd"


# ── Download ───────────────────────────────────────────────────────────
def download_file(url: str, dest: str) -> bool:
    """Download url to dest. Returns True on success."""
    try:
        print(f"  Downloading {dest} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest) / 1024
        print(f"done ({size:.0f} KB)")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False


# ── Preprocessing ──────────────────────────────────────────────────────
def preprocess(df_train: pd.DataFrame,
               df_test:  pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess NSL-KDD DataFrames:
      1. One-hot encode categorical features
      2. MinMax-scale numerical features
      3. Binarise labels: normal=0, attack=1

    Fit scaler and encoder on train only, apply to both.
    """
    def encode_and_scale(df: pd.DataFrame,
                          scaler,
                          train_cats: list[str]) -> pd.DataFrame:
        # One-hot encode
        dummies = pd.get_dummies(df[CAT_COLS], drop_first=False)
        # Cast to uint8 — prevents bool dtype that reads back as object from CSV
        dummies = dummies.astype(np.uint8)
        # Align to train columns (test may lack some service values)
        dummies = dummies.reindex(columns=train_cats, fill_value=0)

        # Scale numerical
        num = pd.DataFrame(
            scaler.transform(df[NUM_COLS]),
            columns=NUM_COLS,
            index=df.index,
        )

        # Binary label
        label = (df['label'] != 'normal').astype(np.int32)
        label.name = 'label'

        return pd.concat([num, dummies, label], axis=1).reset_index(drop=True)

    # Fit on training data only
    scaler = MinMaxScaler()
    scaler.fit(df_train[NUM_COLS])

    train_dummies = pd.get_dummies(df_train[CAT_COLS], drop_first=False)
    train_cat_cols = list(train_dummies.columns)

    df_train_proc = encode_and_scale(df_train, scaler, train_cat_cols)
    df_test_proc  = encode_and_scale(df_test,  scaler, train_cat_cols)

    return df_train_proc, df_test_proc


# ── Main ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true",
                   help="Use local KDDTrain+.txt and KDDTest+.txt "
                        "instead of downloading")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_raw = "KDDTrain+.txt"
    test_raw  = "KDDTest+.txt"

    # ── Obtain raw files ──
    if args.local:
        if not os.path.exists(train_raw) or not os.path.exists(test_raw):
            print("ERROR: KDDTrain+.txt and KDDTest+.txt not found locally.")
            print("Download from https://github.com/defcom17/NSL_KDD")
            return
        print("Using local files.")
    else:
        print("Downloading NSL-KDD from GitHub...")
        ok_train = download_file(TRAIN_URL, train_raw)
        ok_test  = download_file(TEST_URL,  test_raw)
        if not ok_train or not ok_test:
            print("\nAutomatic download failed.")
            print("Manual steps:")
            print("  1. Download KDDTrain+.txt from:")
            print(f"     {TRAIN_URL}")
            print("  2. Download KDDTest+.txt from:")
            print(f"     {TEST_URL}")
            print("  3. Place both files here and run:")
            print("     python prepare_nslkdd.py --local")
            return

    # ── Load ──
    print("\nLoading raw files...")
    df_train = pd.read_csv(train_raw, header=None, names=COLUMNS)
    df_test  = pd.read_csv(test_raw,  header=None, names=COLUMNS)
    print(f"  Train: {len(df_train):,} rows | "
          f"attack ratio: {(df_train['label'] != 'normal').mean():.2f}")
    print(f"  Test : {len(df_test):,}  rows | "
          f"attack ratio: {(df_test['label'] != 'normal').mean():.2f}")

    # ── Preprocess ──
    print("\nPreprocessing...")
    df_train_proc, df_test_proc = preprocess(df_train, df_test)
    n_features = df_train_proc.shape[1] - 1  # exclude label
    print(f"  Feature dimension: {n_features}")
    print(f"  Train processed  : {len(df_train_proc):,} rows")
    print(f"  Test  processed  : {len(df_test_proc):,}  rows")
    print(f"  Label distribution (train): "
          f"normal={( df_train_proc['label']==0).sum():,}  "
          f"attack={(df_train_proc['label']==1).sum():,}")

    # ── Save ──
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out  = os.path.join(OUTPUT_DIR, "test.csv")
    df_train_proc.to_csv(train_out, index=False)
    df_test_proc.to_csv(test_out,   index=False)
    print(f"\nSaved:")
    print(f"  {train_out}")
    print(f"  {test_out}")
    print(f"\nRun the MI experiment with:")
    print(f"  python mi_nslkdd.py --data_dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
