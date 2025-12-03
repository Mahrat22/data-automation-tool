"""
data_automation.py

A small command-line tool that:
- Loads CSV or JSON data
- Cleans it (duplicates, missing values, column names)
- Generates summary reports
- Saves cleaned data + reports to an output folder

Usage example:
    python data_automation.py input.csv --output-dir output
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


import pandas as pd


def load_data(input_path: Path) -> pd.DataFrame:
    """Load CSV or JSON file into a pandas DataFrame."""
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix == ".json":
        # assume JSON array of objects
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (use .csv or .json)")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace spaces with underscores in column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - numeric columns -> median
    - non-numeric -> most frequent value
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in non_numeric_cols:
        if df[col].isna().any():
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
                df[col] = df[col].fillna(mode_val)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform full cleaning pipeline."""
    df = clean_column_names(df)

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    print(f"Removed {before - after} duplicate rows")

    df = handle_missing_values(df)

    return df


def generate_summary_reports(df: pd.DataFrame, output_dir: Path, base_name: str) -> None:
    """
    Create summary reports:
    - numeric_summary.csv  -> describe() for numeric columns
    - full_summary.csv     -> describe(include='all')
    - value_counts_*.csv   -> value counts for each non-numeric column
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_summary_path = output_dir / f"{base_name}_numeric_summary.csv"
    full_summary_path = output_dir / f"{base_name}_full_summary.csv"

    # Numeric summary
    numeric_df = df.describe()
    numeric_df.to_csv(numeric_summary_path)
    print(f"Saved numeric summary to {numeric_summary_path}")

    # Full summary
    full_df = df.describe(include="all")
    full_df.to_csv(full_summary_path)
    print(f"Saved full summary to {full_summary_path}")

    # Value counts for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    for col in non_numeric_cols:
        vc = df[col].value_counts(dropna=False)
        vc_path = output_dir / f"{base_name}_value_counts_{col}.csv"
        vc.to_csv(vc_path, header=["count"])
        print(f"Saved value counts for column '{col}' to {vc_path}")


def apply_optional_filter(df: pd.DataFrame, filter_column: str, min_value: float) -> pd.DataFrame:
    """
    Optionally filter rows where column >= min_value.
    Only applied if filter_column is provided and exists.
    """
    if not filter_column:
        return df

    if filter_column not in df.columns:
        print(f"Warning: filter column '{filter_column}' not found. Skipping filter.")
        return df

    try:
        filtered_df = df[df[filter_column].astype(float) >= min_value]
        print(
            f"Applied filter: {filter_column} >= {min_value}. "
            f"Rows before: {len(df)}, after: {len(filtered_df)}"
        )
        return filtered_df
    except ValueError:
        print(
            f"Warning: could not convert column '{filter_column}' to float. "
            "Skipping filter."
        )
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data Processing & Automation Tool (CSV/JSON)"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input CSV or JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store cleaned data and reports (default: output)",
    )
    parser.add_argument(
        "--filter-column",
        type=str,
        default="",
        help="Optional: numeric column to filter on (e.g., 'price')",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.0,
        help="Optional: minimum value for filter-column (default: 0.0)",
    )
    return parser.parse_args()

def generate_charts(df: pd.DataFrame, output_dir: Path, base_name: str) -> None:
    """
    Generate charts for numeric and categorical data.
    Saves PNG images in the output directory.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Histograms for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].dropna().empty:
            continue

        plt.figure()
        df[col].plot(kind="hist", bins=10)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        chart_path = output_dir / f"{base_name}_hist_{col}.png"
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved histogram for '{col}' to {chart_path}")

    # 2. Bar charts for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    for col in non_numeric_cols:
        vc = df[col].value_counts()
        if vc.empty:
            continue

        plt.figure()
        vc.plot(kind="bar")
        plt.title(f"Value counts of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        chart_path = output_dir / f"{base_name}_bar_{col}.png"
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved bar chart for '{col}' to {chart_path}")

def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print(f"Loading data from {input_path}...")
    df = load_data(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Cleaning data...")
    df_clean = clean_data(df)

    # Optional filter
    df_filtered = apply_optional_filter(
        df_clean, args.filter_column, args.min_value
    )

    base_name = input_path.stem

    # Save cleaned / filtered data
    cleaned_path = output_dir / f"{base_name}_cleaned.csv"
    df_filtered.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned data to {cleaned_path}")

    # Generate summary reports
    generate_summary_reports(df_filtered, output_dir, base_name)
    generate_charts(df_filtered, output_dir, base_name)


    print("Done.")


if __name__ == "__main__":
    main()
