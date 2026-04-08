from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_gambling_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame()

    # 1. Identifiers
    df["player_id"] = np.arange(100000, 100000 + n)
    df["prediction_year"] = 2026
    df["signup_date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1500, n), unit="D"
    )
    df["account_tenure_days"] = (pd.to_datetime("2026-01-01") - df["signup_date"]).dt.days
    df["age"] = rng.integers(18, 70, n)
    df["primary_product_type"] = rng.choice(["sportsbook", "casino"], n)

    # 2. Financial
    df["synthetic_income_monthly"] = rng.normal(5000, 2000, n).clip(1000, 15000)
    df["debt_to_income_ratio"] = rng.uniform(0, 1.5, n)
    df["synthetic_credit_score"] = rng.integers(300, 850, n)
    df["credit_score_velocity_7d"] = rng.integers(-30, 10, n)
    df["liquid_savings_balance"] = rng.uniform(0, 50000, n)
    df["liquid_savings_drawdown_rate"] = rng.uniform(0, 0.5, n)
    df["bank_decline_count_7d"] = rng.poisson(1.5, n)

    # 3. Gambling behavior
    df["total_bets_30d"] = rng.integers(0, 1000, n)
    df["active_days_30d"] = rng.integers(1, 30, n)
    df["avg_bets_per_active_day"] = df["total_bets_30d"] / df["active_days_30d"]
    df["avg_session_duration_minutes"] = rng.uniform(10, 180, n)

    df["total_handle_30d"] = rng.uniform(100, 20000, n)
    df["net_loss_30d"] = df["total_handle_30d"] * rng.uniform(-0.2, 0.5, n)

    df["avg_bet_size"] = df["total_handle_30d"] / (df["total_bets_30d"] + 1)
    df["max_bet_size"] = df["avg_bet_size"] * rng.uniform(2, 10, n)

    df["live_bet_ratio"] = rng.uniform(0, 1, n)
    df["parlay_ratio"] = rng.uniform(0, 1, n)

    # 4. Deposits
    df["deposit_count_30d"] = rng.poisson(10, n)
    df["deposit_amount_30d"] = df["deposit_count_30d"] * rng.uniform(20, 200, n)
    df["avg_deposit_size"] = df["deposit_amount_30d"] / (df["deposit_count_30d"] + 1)

    df["deposit_growth_rate_4w"] = rng.uniform(-0.5, 2, n)
    df["failed_deposit_count_30d"] = rng.poisson(2, n)

    df["withdrawal_count_30d"] = rng.poisson(5, n)
    df["withdrawal_amount_30d"] = rng.uniform(0, 10000, n)

    df["withdrawal_reversal_rate"] = rng.uniform(0, 1, n)
    df["rapid_redeposit_rate"] = rng.uniform(0, 1, n)

    # 5. Micro signals
    df["session_velocity_index"] = rng.uniform(0, 3, n)
    df["mean_inter_session_gap_hours"] = rng.uniform(1, 100, n)
    df["inter_session_gap_trend"] = rng.uniform(-2, 2, n)

    df["night_time_betting_ratio"] = rng.uniform(0, 1, n)
    df["weekend_betting_ratio"] = rng.uniform(0, 1, n)

    df["same_day_multiple_session_flag"] = rng.choice([0, 1], n)

    df["post_loss_session_return_time_hours"] = rng.uniform(0, 24, n)
    df["chase_magnitude_index"] = rng.uniform(0, 3, n)
    df["loss_streak_length_max"] = rng.integers(0, 20, n)

    df["volatility_of_daily_losses"] = rng.uniform(0, 500, n)

    # 6. Marketing
    df["bonus_offer_count_30d"] = rng.integers(0, 20, n)
    df["bonus_redemption_rate"] = rng.uniform(0, 1, n)
    df["bonus_sensitivity_index"] = rng.uniform(0, 1, n)

    df["push_notification_density"] = rng.integers(0, 50, n)
    df["gambling_ad_ctr"] = rng.uniform(0, 1, n)

    df["affiliate_referral_source"] = rng.choice(["organic", "influencer", "vip"], n)
    df["vip_status"] = rng.choice([0, 1], n, p=[0.9, 0.1])

    # 7. Regulation
    df["limit_setting_flag"] = rng.choice([0, 1], n)
    df["deposit_limit_breach_attempts"] = rng.poisson(2, n)
    df["time_out_requests_count"] = rng.poisson(1, n)

    df["responsible_gambling_message_count"] = rng.integers(0, 20, n)
    df["regulatory_friction_score"] = rng.uniform(0, 1, n)
    df["friction_bypass_attempt_flag"] = rng.choice([0, 1], n)

    # 8. Target (logic-based)
    risk_score = (
        0.3 * df["chase_magnitude_index"]
        + 0.3 * df["withdrawal_reversal_rate"]
        + 0.2 * df["bank_decline_count_7d"]
        + 0.2 * df["night_time_betting_ratio"]
    )
    df["future_high_risk_flag"] = (risk_score > np.percentile(risk_score, 85)).astype(int)

    # 9. Make data messy, but keep id and target intact for modeling
    protected_cols = {"player_id", "prediction_year", "future_high_risk_flag"}
    for col in df.columns:
        if col in protected_cols:
            continue
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    # Add a bit of outlier noise
    age_noise_mask = rng.random(n) < 0.02
    df.loc[age_noise_mask, "age"] = df.loc[age_noise_mask, "age"] + rng.integers(-10, 10, age_noise_mask.sum())
    df["age"] = df["age"].clip(18, 90)

    deposit_spike_mask = rng.random(n) < 0.02
    df.loc[deposit_spike_mask, "deposit_amount_30d"] = (
        df.loc[deposit_spike_mask, "deposit_amount_30d"] * 5
    )

    return df


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_gambling_data.csv"

    df = generate_synthetic_gambling_data()
    df.to_csv(output_file, index=False)
    print(f"Wrote {len(df):,} rows to {output_file}")


if __name__ == "__main__":
    main()
