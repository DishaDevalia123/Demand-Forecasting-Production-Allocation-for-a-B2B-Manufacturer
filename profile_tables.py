"""
profile_tables.py — first-pass profiling of the raw generated CSVs.

Usage:
    python profile_tables.py /path/to/csv_folder
    python profile_tables.py /path/to/csv_folder > profile_report.txt

Reads every .csv in the folder TWICE:
  1. with pandas type inference  -> what pandas THINKS each column is
  2. as raw strings              -> what is ACTUALLY written in the file

Nothing is cleaned, dropped, filled or converted here.
This script only describes what is on disk.
"""

import sys
import glob
import os
from collections import defaultdict

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

# text values that usually MEAN missing but are not read as null by default
NULL_LIKE = {
    "", " ", "na", "n/a", "n.a.", "null", "none", "nan", "nil",
    "-", "--", "?", "unknown", "unk", "missing", "tbd", "#n/a",
}

LINE = "=" * 90
SUB = "-" * 90


def human(n):
    return f"{n:,}"


def profile_file(path):
    name = os.path.basename(path)
    print(LINE)
    print(f"TABLE: {name}")
    print(LINE)

    df = pd.read_csv(path, low_memory=False)                                # inferred
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)  # literal

    n_rows, n_cols = df.shape
    print(f"\nSHAPE: {human(n_rows)} rows x {n_cols} columns")
    print(f"FULLY DUPLICATE ROWS: {human(df.duplicated().sum())}")

    # ---------- per-column summary ----------
    print(f"\n{SUB}\nCOLUMN SUMMARY\n{SUB}")
    rows = []
    for c in df.columns:
        s, r = df[c], raw[c]
        nulls = int(s.isna().sum())
        nulllike = int(r.str.strip().str.lower().isin(NULL_LIKE).sum())
        nuniq = int(s.nunique(dropna=True))
        pad_ws = int((r != r.str.strip()).sum())
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "nulls": nulls,
            "null_%": round(100 * nulls / n_rows, 2) if n_rows else 0,
            "nulllike_txt": nulllike,
            "n_unique": nuniq,
            "uniq_%": round(100 * nuniq / n_rows, 2) if n_rows else 0,
            "pad_ws": pad_ws,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- candidate keys ----------
    print(f"\n{SUB}\nCANDIDATE KEYS (single columns that are unique AND complete)\n{SUB}")
    keys = [c for c in df.columns if df[c].notna().all() and df[c].nunique() == n_rows]
    print(keys if keys else "none - the grain is defined by a COMBINATION of columns")

    # ---------- value detail ----------
    print(f"\n{SUB}\nVALUES  (full counts if <= 25 distinct, else samples + ranges)\n{SUB}")
    for c in df.columns:
        s = df[c]
        nuniq = int(s.nunique(dropna=True))
        print(f"\n  {c}  [{s.dtype}, {human(nuniq)} distinct]")

        if nuniq <= 25:
            for val, cnt in raw[c].value_counts(dropna=False).items():
                shown = repr(val) if (val.strip() != val or val == "") else val
                print(f"      {shown:<40} {human(int(cnt))}")
            continue

        print(f"      sample: {list(s.dropna().unique()[:5])}")

        if pd.api.types.is_numeric_dtype(s):
            print(f"      min={s.min()}  max={s.max()}  "
                  f"mean={round(float(s.mean()), 3) if s.notna().any() else 'NA'}")
            print(f"      negatives={human(int((s < 0).sum()))}  "
                  f"zeros={human(int((s == 0).sum()))}")
        else:
            try:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            except Exception:
                parsed = pd.to_datetime(s, errors="coerce")
            ok = int(parsed.notna().sum())
            if ok > 0.5 * int(s.notna().sum()):
                print(f"      looks like a date: min={parsed.min()}  max={parsed.max()}")
                print(f"      unparseable: {human(int(s.notna().sum()) - ok)}")
            else:
                lens = raw[c].str.len()
                print(f"      text length min={lens.min()} max={lens.max()}")
                if s.dtype == object:
                    lower_uniq = int(s.dropna().astype(str).str.lower().nunique())
                    if lower_uniq != nuniq:
                        print(f"      CASE VARIANTS: {nuniq} distinct raw vs "
                              f"{lower_uniq} distinct lowercased")
    print()
    return name, set(df.columns)


def main():
    if len(sys.argv) < 2:
        print("usage: python profile_tables.py <folder-with-csvs>")
        sys.exit(1)

    folder = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        print(f"no CSVs found in {folder}")
        sys.exit(1)

    print(f"Found {len(paths)} CSV files in {folder}\n")

    seen = []
    for p in paths:
        try:
            seen.append(profile_file(p))
        except Exception as e:
            print(f"FAILED to profile {p}: {type(e).__name__}: {e}\n")

    print(LINE)
    print("SHARED COLUMN NAMES (candidate join keys)")
    print(LINE)
    where = defaultdict(list)
    for name, cols in seen:
        for c in cols:
            where[c].append(name)
    shared = {c: t for c, t in where.items() if len(t) > 1}
    if not shared:
        print("no column name appears in more than one table")
    for c, tables in sorted(shared.items()):
        print(f"  {c:<30} {tables}")


if __name__ == "__main__":
    main()