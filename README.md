# Data Automation Tool

A Python command-line tool that:

- Loads CSV or JSON data
- Cleans it (duplicate removal, missing values, column name normalization)
- Generates summary reports (numeric + full stats)
- Computes value counts for categorical columns
- Creates charts (histograms and bar charts) as PNG images

## Usage

```bash
python data_automation.py input.csv --output-dir output
