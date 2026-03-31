import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Split labeled data into train/val/test sets")
parser.add_argument("input", help="Path to labeled CSV")
parser.add_argument("--train", type=float, default=0.8, help="Train fraction (default: 0.8)")
parser.add_argument("--val", type=float, default=0.1, help="Val fraction (default: 0.1)")
parser.add_argument("--out-dir", default="assets", help="Output directory (default: assets)")
args = parser.parse_args()

df = pd.read_csv(args.input)
n = len(df)

train_end = int(n * args.train)
val_end = int(n * (args.train + args.val))

splits = {
    "train": df.iloc[:train_end],
    "val":   df.iloc[train_end:val_end],
    "test":  df.iloc[val_end:],
}

for name, split in splits.items():
    path = f"{args.out_dir}/{name}.csv"
    split.to_csv(path, index=False)
    hard = (split["Answer"] >= 0.5).astype(int)
    print(f"{name}: {len(split)} rows | 0: {(hard==0).sum()} ({(hard==0).mean()*100:.1f}%) | 1: {(hard==1).sum()} ({(hard==1).mean()*100:.1f}%) -> {path}")
