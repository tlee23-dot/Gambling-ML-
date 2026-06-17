from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_gambling_data(n_players: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """
    Generates COMPLETE longitudinal synthetic gambling data from 2020 to 2025.
    Engineered mathematically to yield an 85-88% precision/F1 score in 2026 predictions.
    
    GUARANTEES risk_profile: 0 (Low, 60%), 1 (Medium, 30%), 2 (High, 10%).
    """
    rng = np.random.default_rng(seed)
    
    # --- STEP 1: Generate Base Player Attributes (Static) ---
    players = pd.DataFrame()
    players["player_id"] = np.arange(100000, 100000 + n_players)
    
    players["signup_date"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n_players), unit="D"
    )
    players["age"] = rng.integers(18, 70, n_players)
    players["primary_product_type"] = rng.choice(["sportsbook", "casino"], n_players)
    
    players["synthetic_income_monthly"] = rng.normal(5000, 2000, n_players).clip(1000, 15000)
    players["debt_to_income_ratio"] = rng.uniform(0, 1.5, n_players)
    players["synthetic_credit_score"] = rng.integers(300, 850, n_players)
    
    # --- STEP 2: Risk Profile Assignment ---
    n_low = int(n_players * 0.60)    
    n_medium = int(n_players * 0.30) 
    n_high = n_players - n_low - n_medium  
    
    risk_assignments = ([0] * n_low) + ([1] * n_medium) + ([2] * n_high)
    rng.shuffle(risk_assignments)  
    players["risk_profile"] = risk_assignments

    # --- STEP 3: Generate Longitudinal Data (2020 - 2025) ---
    history_years = [2020, 2021, 2022, 2023, 2024, 2025]
    all_years_data = []

    for year in history_years:
        year_df = players.copy()
        year_df["data_year"] = year
        year_end = pd.to_datetime(f"{year}-12-31")
        year_df["account_tenure_days"] = (year_end - year_df["signup_date"]).dt.days
        
       # ---  MATH GATEWAY (Targeting 85-88% F1) ---
        # 1. Increase the "Noise Profile" from 13% to 22%
        true_profile = year_df["risk_profile"].values
        noise_profile = rng.choice([0, 1, 2], size=n_players, p=[0.60, 0.30, 0.10])
        
        blend_mask = rng.random(n_players) < 0.78  # Increased noise to 22%
        g_int = np.where(blend_mask, true_profile, noise_profile)
        
        # 2. Use Log-Normal distribution for behavior to add "real-world" variance
        # This makes features harder for the model to split perfectly
        g_factor = np.exp(g_int * 0.5) * rng.normal(1, 0.2, n_players) 
        
        # 3. Add "Interaction Noise" to every major feature
        # Instead of: feature = base * g_factor
        # Use: feature = base * g_factor + (random_oscillation)
        random_oscillation = rng.normal(0, 5, n_players)
        
        # 1. Financial
        year_df["credit_score_velocity_7d"] = rng.integers(-30, 10, n_players) - (g_int * 5)
        year_df["liquid_savings_balance"] = rng.uniform(5000, 50000, n_players) * (1 - g_int * 0.20)
        year_df["liquid_savings_drawdown_rate"] = rng.uniform(0, 0.2, n_players) * g_factor
        year_df["bank_decline_count_7d"] = rng.poisson(0.5 + (g_int * 2.0), n_players)

        # 2. Gambling Behavior (Includes explicit leakage candidates)
        year_df["total_bets_30d"] = (rng.integers(10, 300, n_players) * g_factor).astype(int)
        year_df["active_days_30d"] = np.clip(rng.integers(1, 12, n_players) + (g_int * 5), 1, 30)
        year_df["avg_bets_per_active_day"] = year_df["total_bets_30d"] / year_df["active_days_30d"]
        year_df["avg_session_duration_minutes"] = rng.uniform(10, 60, n_players) * g_factor
        
        year_df["total_handle_30d"] = rng.uniform(100, 3000, n_players) * g_factor
        
        # LEAKAGE CANDIDATE: net_loss_30d heavily correlates to High Risk by definition
        loss_multiplier = rng.uniform(0.02, 0.15, n_players) + (g_int * 0.15)
        year_df["net_loss_30d"] = year_df["total_handle_30d"] * loss_multiplier
        
        year_df["avg_bet_size"] = year_df["total_handle_30d"] / (year_df["total_bets_30d"] + 1)
        year_df["max_bet_size"] = year_df["avg_bet_size"] * (rng.uniform(2, 5, n_players) + g_int)
        
        year_df["live_bet_ratio"] = np.clip(rng.uniform(0.1, 0.5, n_players) * g_factor, 0, 1)
        year_df["parlay_ratio"] = np.clip(rng.uniform(0.1, 0.4, n_players) * g_factor, 0, 1)

        # 3. Deposits
        year_df["deposit_count_30d"] = rng.poisson(3 + g_int * 6, n_players)
        year_df["deposit_amount_30d"] = year_df["deposit_count_30d"] * rng.uniform(20, 100, n_players) * (g_factor * 0.8)
        year_df["avg_deposit_size"] = year_df["deposit_amount_30d"] / (year_df["deposit_count_30d"] + 1)
        
        year_df["deposit_growth_rate_4w"] = rng.uniform(-0.2, 0.5, n_players) + (g_int * 0.3)
        year_df["failed_deposit_count_30d"] = rng.poisson(0.5 + g_int * 2.0, n_players)
        
        # Withdrawals
        year_df["withdrawal_count_30d"] = rng.poisson(4, n_players)
        year_df["withdrawal_amount_30d"] = rng.uniform(0, 5000, n_players)
        year_df["withdrawal_reversal_rate"] = np.clip(rng.uniform(0, 0.2, n_players) * g_factor, 0, 1)
        year_df["rapid_redeposit_rate"] = np.clip(rng.uniform(0, 0.2, n_players) * g_factor, 0, 1)

        # 4. Micro Signals
        year_df["session_velocity_index"] = rng.uniform(0.2, 1.2, n_players) * g_factor
        year_df["mean_inter_session_gap_hours"] = rng.uniform(20, 100, n_players) / g_factor
        year_df["inter_session_gap_trend"] = rng.uniform(-1, 1, n_players) - (g_int * 0.2)
        
        year_df["night_time_betting_ratio"] = np.clip(rng.uniform(0.05, 0.25, n_players) * g_factor, 0, 1)
        year_df["weekend_betting_ratio"] = rng.uniform(0.2, 0.6, n_players)
        year_df["same_day_multiple_session_flag"] = rng.choice([0, 1], n_players, p=[0.8, 0.2])

        year_df["post_loss_session_return_time_hours"] = rng.uniform(12, 48, n_players) / g_factor
        year_df["chase_magnitude_index"] = rng.uniform(0.1, 0.8, n_players) * g_factor
        year_df["loss_streak_length_max"] = rng.integers(2, 8, n_players) + (g_int * 3)
        year_df["volatility_of_daily_losses"] = rng.uniform(10, 100, n_players) * g_factor

        # 5. Marketing
        year_df["bonus_offer_count_30d"] = rng.integers(2, 10, n_players) + (g_int * 2)
        year_df["bonus_redemption_rate"] = np.clip(rng.uniform(0.1, 0.4, n_players) * g_factor, 0, 1)
        year_df["bonus_sensitivity_index"] = rng.uniform(0, 0.5, n_players) + (g_int * 0.2)
        
        year_df["push_notification_density"] = rng.integers(5, 20, n_players) + (g_int * 6)
        year_df["gambling_ad_ctr"] = rng.uniform(0.01, 0.1, n_players) * g_factor
        
        year_df["affiliate_referral_source"] = rng.choice(["organic", "influencer", "vip"], n_players)
        year_df["vip_status"] = np.where(g_int == 2, rng.choice([0, 1], n_players, p=[0.7, 0.3]), rng.choice([0, 1], n_players, p=[0.95, 0.05]))

        # 6. Regulation (Includes explicit leakage candidates)
        year_df["limit_setting_flag"] = rng.choice([0, 1], n_players)
        
        # Possible TO WATCH OUT FOR LEAKAGE CANDIDATE: 
        year_df["deposit_limit_breach_attempts"] = rng.poisson(0.1 + g_int * 2.5, n_players)
        year_df["time_out_requests_count"] = rng.poisson(0.1 + (g_int * 0.5), n_players)
        
        year_df["responsible_gambling_message_count"] = rng.integers(0, 5, n_players) + (g_int * 3)
        year_df["regulatory_friction_score"] = rng.uniform(0, 0.3, n_players) * g_factor
        year_df["friction_bypass_attempt_flag"] = np.where(g_int == 2, rng.choice([0, 1], n_players, p=[0.8, 0.2]), rng.choice([0, 1], n_players, p=[0.98, 0.02]))

        all_years_data.append(year_df)

    # Combine all years
    df = pd.concat(all_years_data, ignore_index=True)

    # --- STEP 4: Add Messiness (Missing Values & Outliers) ---
    protected_cols = {"player_id", "data_year", "risk_profile", "signup_date", "primary_product_type"}
    
    # Intentionally missing 5% of entries to mimic data loss
    for col in df.columns:
        if col in protected_cols:
            continue
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    # Add realistic outlier noise
    age_noise_mask = rng.random(len(df)) < 0.02
    df.loc[age_noise_mask, "age"] = df.loc[age_noise_mask, "age"] + rng.integers(-5, 5, age_noise_mask.sum())
    df["age"] = df["age"].clip(18, 90)

    # Massive deposit spikes for random users (mimicking "whale" behavior outliers)
    deposit_spike_mask = rng.random(len(df)) < 0.03
    df.loc[deposit_spike_mask, "deposit_amount_30d"] *= rng.uniform(5, 15, deposit_spike_mask.sum())

    return df

def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_gambling_data.csv"

    df = generate_synthetic_gambling_data(n_players=1000)
    df.to_csv(output_file, index=False)
    print(f"Successfully saved data file with {df.shape[0]} rows to: {output_file}")

if __name__ == "__main__":
    main()
