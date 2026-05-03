from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_gambling_data(n_players: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """
    Generates COMPLETE longitudinal synthetic gambling data from 2020 to 2025 
    to predict risk profiles for 2026.
    
    GUARANTEES risk_profile 0, 1, 2 are all present with controlled distribution:
    - 0 (Low): 60%
    - 1 (Medium): 30% 
    - 2 (High): 10%
    """
    rng = np.random.default_rng(seed)
    
    # --- STEP 1: Generate Base Player Attributes (Static) ---
    players = pd.DataFrame()
    players["player_id"] = np.arange(100000, 100000 + n_players)
    
    # Signup dates between 2018 and 2020 so they are active during 2020-2025
    players["signup_date"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n_players), unit="D"
    )
    players["age"] = rng.integers(18, 70, n_players)
    players["primary_product_type"] = rng.choice(["sportsbook", "casino"], n_players)
    
    # Static financial background
    players["synthetic_income_monthly"] = rng.normal(5000, 2000, n_players).clip(1000, 15000)
    players["debt_to_income_ratio"] = rng.uniform(0, 1.5, n_players)
    players["synthetic_credit_score"] = rng.integers(300, 850, n_players)
    
    # --- STEP 2: GUARANTEE Risk Profiles 0, 1, 2 with EXACT distribution ---
    # Explicitly assign to ensure all categories exist
    n_low = int(n_players * 0.60)    # 60% Low Risk (0)
    n_medium = int(n_players * 0.30) # 30% Medium Risk (1)
    n_high = n_players - n_low - n_medium  # 10% High Risk (2)
    
    risk_assignments = ([0] * n_low) + ([1] * n_medium) + ([2] * n_high)
    rng.shuffle(risk_assignments)  # Shuffle to randomize assignment
    players["risk_profile"] = risk_assignments
    
    print(f" Risk Profile Distribution:")
    print(f"   0 (Low): {n_low} players ({n_low/n_players*100:.1f}%)")
    print(f"   1 (Medium): {n_medium} players ({n_medium/n_players*100:.1f}%)")
    print(f"   2 (High): {n_high} players ({n_high/n_players*100:.1f}%)")

    # --- STEP 3: Generate COMPLETE Longitudinal Data (2020 - 2025) ---
    history_years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_years_data = []

    for year in history_years:
        year_df = players.copy()
        year_df["data_year"] = year
        year_end = pd.to_datetime(f"{year}-12-31")
        year_df["account_tenure_days"] = (year_end - year_df["signup_date"]).dt.days
        
        # Risk factor scales ALL behaviors
        risk_factor = year_df["risk_profile"].map({0: 1.0, 1: 1.5, 2: 2.5})
        
        # 1. Financial (Yearly updates)
        year_df["credit_score_velocity_7d"] = rng.integers(-30, 10, n_players)
        year_df["liquid_savings_balance"] = rng.uniform(0, 50000, n_players) * (1 - year_df["risk_profile"] * 0.1)
        year_df["liquid_savings_drawdown_rate"] = rng.uniform(0, 0.5, n_players) * risk_factor
        year_df["bank_decline_count_7d"] = rng.poisson(1.5 + year_df["risk_profile"], n_players)

        # 2. Gambling Behavior (ALL original metrics)
        year_df["total_bets_30d"] = rng.integers(10, 1000, n_players) * risk_factor
        year_df["active_days_30d"] = rng.integers(1, 30, n_players)
        year_df["avg_bets_per_active_day"] = year_df["total_bets_30d"] / year_df["active_days_30d"]
        year_df["avg_session_duration_minutes"] = rng.uniform(10, 180, n_players)
        
        year_df["total_handle_30d"] = rng.uniform(100, 20000, n_players) * risk_factor
        loss_multiplier = rng.uniform(-0.2, 0.5, n_players) - (year_df["risk_profile"] * 0.1)
        year_df["net_loss_30d"] = year_df["total_handle_30d"] * loss_multiplier
        
        year_df["avg_bet_size"] = year_df["total_handle_30d"] / (year_df["total_bets_30d"] + 1)
        year_df["max_bet_size"] = year_df["avg_bet_size"] * rng.uniform(2, 10, n_players)
        
        year_df["live_bet_ratio"] = rng.uniform(0, 1, n_players)
        year_df["parlay_ratio"] = rng.uniform(0, 1, n_players)

        # 3. Deposits (ALL original metrics)
        year_df["deposit_count_30d"] = rng.poisson(10 + year_df["risk_profile"] * 5, n_players)
        year_df["deposit_amount_30d"] = year_df["deposit_count_30d"] * rng.uniform(20, 200, n_players)
        year_df["avg_deposit_size"] = year_df["deposit_amount_30d"] / (year_df["deposit_count_30d"] + 1)
        
        year_df["deposit_growth_rate_4w"] = rng.uniform(-0.5, 2, n_players)
        year_df["failed_deposit_count_30d"] = rng.poisson(2 + year_df["risk_profile"] * 2, n_players)
        
        year_df["withdrawal_count_30d"] = rng.poisson(5, n_players)
        year_df["withdrawal_amount_30d"] = rng.uniform(0, 10000, n_players)
        year_df["withdrawal_reversal_rate"] = rng.uniform(0, 1, n_players) * risk_factor
        year_df["rapid_redeposit_rate"] = rng.uniform(0, 1, n_players) * risk_factor

        # 4. Micro Signals (ALL original metrics)
        year_df["session_velocity_index"] = rng.uniform(0, 3, n_players) * risk_factor
        year_df["mean_inter_session_gap_hours"] = rng.uniform(1, 100, n_players)
        year_df["inter_session_gap_trend"] = rng.uniform(-2, 2, n_players)
        
        year_df["night_time_betting_ratio"] = rng.uniform(0, 1, n_players) * risk_factor
        year_df["weekend_betting_ratio"] = rng.uniform(0, 1, n_players)
        year_df["same_day_multiple_session_flag"] = rng.choice([0, 1], n_players, p=[0.7, 0.3])
        
        year_df["post_loss_session_return_time_hours"] = rng.uniform(0, 24, n_players)
        year_df["chase_magnitude_index"] = rng.uniform(0, 3, n_players) * risk_factor
        year_df["loss_streak_length_max"] = rng.integers(0, 20, n_players)
        year_df["volatility_of_daily_losses"] = rng.uniform(0, 500, n_players) * risk_factor

        # 5. Marketing (ALL original metrics)
        year_df["bonus_offer_count_30d"] = rng.integers(0, 20, n_players)
        year_df["bonus_redemption_rate"] = rng.uniform(0, 1, n_players) * risk_factor
        year_df["bonus_sensitivity_index"] = rng.uniform(0, 1, n_players)
        
        year_df["push_notification_density"] = rng.integers(0, 50, n_players)
        year_df["gambling_ad_ctr"] = rng.uniform(0, 1, n_players)
        
        year_df["affiliate_referral_source"] = rng.choice(["organic", "influencer", "vip"], n_players)
        year_df["vip_status"] = rng.choice([0, 1], n_players, p=[0.9, 0.1])

        # 6. Regulation (ALL original metrics)
        year_df["limit_setting_flag"] = rng.choice([0, 1], n_players)
        year_df["deposit_limit_breach_attempts"] = rng.poisson(2 + year_df["risk_profile"], n_players)
        year_df["time_out_requests_count"] = rng.poisson(1, n_players)
        
        year_df["responsible_gambling_message_count"] = rng.integers(0, 20, n_players)
        year_df["regulatory_friction_score"] = rng.uniform(0, 1, n_players)
        year_df["friction_bypass_attempt_flag"] = rng.choice([0, 1], n_players, p=[0.95, 0.05])

        all_years_data.append(year_df)

    # Combine all years
    df = pd.concat(all_years_data, ignore_index=True)

    # --- STEP 4: Add Messiness (Missing values & Outliers) ---
    protected_cols = {"player_id", "data_year", "risk_profile", "signup_date", "primary_product_type"}
    
    # ~5% missing values
    for col in df.columns:
        if col in protected_cols:
            continue
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    # Outlier noise
    age_noise_mask = rng.random(len(df)) < 0.02
    df.loc[age_noise_mask, "age"] = df.loc[age_noise_mask, "age"] + rng.integers(-10, 10, age_noise_mask.sum())
    df["age"] = df["age"].clip(18, 90)

    deposit_spike_mask = rng.random(len(df)) < 0.02
    df.loc[deposit_spike_mask, "deposit_amount_30d"] = (
        df.loc[deposit_spike_mask, "deposit_amount_30d"] * 5
    )

    return df


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_gambling_data.csv"

    df = generate_synthetic_gambling_data(n_players=1000)
    
    df.to_csv(output_file, index=False)
    


if __name__ == "__main__":
    main()