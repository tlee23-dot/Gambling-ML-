# Synthetic Dataset Documentation

## Purpose

`synthetic_gambling_data.csv` is a synthetic training dataset for early ML prototyping.
It contains player-level features across behavior, financial activity, marketing exposure,
and regulation interaction.

## Generation

- Script: `src/data/generate_synthetic_data.py`
- Default rows: `10,000`
- Default seed: `42`
- Output: `data/raw/synthetic_gambling_data.csv`

Run:

`python3 src/data/generate_synthetic_data.py`

## Target Variable

- Column: `future_high_risk_flag`
- Type: binary (`0` or `1`)
- Logic: derived from a weighted risk score using:
  - `chase_magnitude_index`
  - `withdrawal_reversal_rate`
  - `bank_decline_count_7d`
  - `night_time_betting_ratio`
- Positive label threshold: top 15% percentile of the risk score.

## Data Quality Characteristics

- Around 5% missing values are injected for most feature columns.
- Identifier and label columns are protected from null injection:
  - `player_id`
  - `prediction_year`
  - `future_high_risk_flag`
- Additional outlier noise is introduced for realism:
  - noisy `age` for ~2% rows (clipped to valid range)
  - `deposit_amount_30d` spikes for ~2% rows

## Modeling Notes

- This is synthetic data; patterns are engineered and not representative of real users.
- Build robust preprocessing:
  - missing value handling
  - numeric scaling (if required by model choice)
  - categorical encoding for:
    - `primary_product_type`
    - `affiliate_referral_source`
