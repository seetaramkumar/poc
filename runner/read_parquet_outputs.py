"""
Read every Parquet output file under runner/output.

Run from the project root:
    python runner/read_parquet_outputs.py

Useful options:
    python runner/read_parquet_outputs.py --rows 20
    python runner/read_parquet_outputs.py --output-dir runner/output/quality
    python runner/read_parquet_outputs.py --show-all-columns
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runner" / "output"


def read_parquet_files(output_dir: Path, rows: int, show_all_columns: bool) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pandas.\n"
            "Install the project dependencies first, for example:\n"
            "  pip install -r stock_regime/requirements.txt"
        ) from exc

    parquet_files = sorted(output_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"No Parquet files found under: {output_dir}")
        return

    if show_all_columns:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)

    print(f"Found {len(parquet_files)} Parquet file(s) under {output_dir}\n")

    for file_path in parquet_files:
        relative_path = file_path.relative_to(PROJECT_ROOT)
        print("=" * 100)
        print(f"File: {relative_path}")

        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            print(f"Could not read file: {exc}")
            continue

        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")

        if df.empty:
            print("(empty dataframe)")
        else:
            print(f"\nFirst {min(rows, len(df))} row(s):")
            print(df.head(rows).to_string(index=False))

        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively read and preview Parquet files from runner/output."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to search recursively. Default: runner/output",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to print per Parquet file. Default: 10",
    )
    parser.add_argument(
        "--show-all-columns",
        action="store_true",
        help="Do not truncate dataframe columns in the terminal output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    read_parquet_files(
        output_dir=output_dir.resolve(),
        rows=max(args.rows, 0),
        show_all_columns=args.show_all_columns,
    )


if __name__ == "__main__":
    main()
