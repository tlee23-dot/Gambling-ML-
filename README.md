# Gambling ML

Clean starter structure for building a gambling-risk ML pipeline.

## Project Structure

- `src/` - source code
- `src/data/generate_synthetic_data.py` - synthetic dataset generator
- `data/raw/` - generated raw datasets
- `docs/` - project and data documentation

## Quick Start

1. Create a virtual environment (recommended):
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Generate synthetic dataset:
   - `python3 src/data/generate_synthetic_data.py`

The generated file will be written to:
- `data/raw/synthetic_gambling_data.csv`

## Notes

- Data is synthetic and intended for model prototyping only.
- The generator uses a fixed random seed by default for reproducibility.