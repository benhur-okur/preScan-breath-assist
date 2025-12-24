# tools/subject_split_loso.py
import argparse
import pandas as pd

'''
python tools/subject_split_loso.py --manifest data/windows/manifest.csv --out data/windows/manifest_loso.csv
'''

def make_loso_splits(manifest_csv: str, out_csv: str, val_strategy: str = "next") -> None:
    df = pd.read_csv(manifest_csv)
    persons = sorted(df["person"].unique().tolist())
    if len(persons) < 3:
        raise RuntimeError("Need at least 3 subjects for LOSO with val+test.")

    # fold_id per test subject
    out_rows = []

    for i, test_person in enumerate(persons):
        remaining = [p for p in persons if p != test_person]

        if val_strategy == "next":
            # deterministic: val = next subject in sorted list (cyclic) among remaining
            idx = (i + 1) % len(persons)
            val_candidate = persons[idx]
            if val_candidate == test_person:
                val_candidate = remaining[0]
            val_person = val_candidate
        else:
            # fallback: first remaining
            val_person = remaining[0]

        train_persons = [p for p in persons if p not in [test_person, val_person]]

        fold_df = df.copy()
        fold_df["fold_id"] = f"fold_{i+1}_test_{test_person}"

        fold_df.loc[fold_df["person"].isin(train_persons), "split"] = "train"
        fold_df.loc[fold_df["person"] == val_person, "split"] = "val"
        fold_df.loc[fold_df["person"] == test_person, "split"] = "test"

        out_rows.append(fold_df)

    out_all = pd.concat(out_rows, ignore_index=True)
    out_all.to_csv(out_csv, index=False)

    print("=== LOSO SPLIT DONE ===")
    print("Input :", manifest_csv)
    print("Output:", out_csv)
    print("Subjects:", persons)
    print("Rows:", len(out_all))
    print("Folds:", out_all["fold_id"].nunique())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/windows/manifest_loso.csv")
    ap.add_argument("--val-strategy", choices=["next", "first"], default="next")
    args = ap.parse_args()
    make_loso_splits(args.manifest, args.out, args.val_strategy)
