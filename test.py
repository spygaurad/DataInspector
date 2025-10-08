#!/usr/bin/env python3
import argparse

import pandas as pd


def is_datetime_like(series: pd.Series, min_samples: int = 5, threshold: float = 0.6) -> bool:
    """
    Return True if `series` looks like datetimes.
    - Tries pandas' datetime parser.
    - If >= `threshold` of non-null values successfully parse (and there are at least `min_samples`),
      we treat the column as datetime-like.
    """
    s = series.dropna().astype(str).str.strip()
    if len(s) < min_samples:
        return False

    parsed = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
    if parsed.notna().mean() >= threshold:
        return True

    # Also treat native datetime dtypes as datetime-like
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_datetime64tz_dtype(series)

def drop_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    dt_cols = [c for c in df.columns if is_datetime_like(df[c])]
    return df.drop(columns=dt_cols), dt_cols

def main():
    ap = argparse.ArgumentParser(description="Remove datetime-like columns from a CSV.")
    ap.add_argument("input_csv", help="Path to input CSV")
    ap.add_argument("output_csv", help="Path to output CSV (datetime columns removed)")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Fraction of parseable values to consider a column datetime-like (default: 0.6)")
    ap.add_argument("--min-samples", type=int, default=5,
                    help="Minimum non-null values required to evaluate a column (default: 5)")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    # Allow user-tuned detection
    def _is_dt(col):
        return is_datetime_like(df[col], min_samples=args.min_samples, threshold=args.threshold)

    dt_cols = [c for c in df.columns if _is_dt(c)]
    df_out = df.drop(columns=dt_cols)
    df_out.to_csv(args.output_csv, index=False)

    print(f"Removed {len(dt_cols)} datetime-like columns: {dt_cols}")
    print(f"Wrote cleaned CSV to: {args.output_csv}")

if __name__ == "__main__":
    main()
