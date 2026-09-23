import json
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, is_train: bool = True, bin_risk_map: dict = None):
    """Compute offline feature matrix with Bayesian BIN risk smoothing and left-closed rolling windows."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["amount_log"] = np.log1p(df["amount_cents"] / 100.0).astype(np.float32)
    df["is_round_amount"] = (df["amount_cents"] % 1000 == 0).astype(np.float32)
    df["amount_cents_mod_100"] = (df["amount_cents"] % 100).astype(np.float32)

    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24.0).astype(np.float32)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.float32)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.float32)

    df["cvv_match_num"] = df["cvv_match"].astype(np.float32)
    df["is_3ds_passed_num"] = df["is_3ds_passed"].astype(np.float32)
    df["risk_no_3ds_cvv_fail"] = ((~df["cvv_match"]) & (~df["is_3ds_passed"])).astype(np.float32)

    df["currency_code"] = (df["currency"] == "USD").astype(np.float32)
    df["billing_country_code"] = (df["billing_country"] == "US").astype(np.float32)

    account_age_days = (df["timestamp"].astype("int64") // 10**9 - df["user_created_at"]) / 86400.0
    df["account_age_days"] = account_age_days.astype(np.float32)
    df["account_age_days_log"] = np.log1p(np.maximum(0, account_age_days)).astype(np.float32)

    if is_train:
        global_mean = df["is_chargeback"].mean()
        bin_stats = df.groupby("card_bin")["is_chargeback"].agg(["count", "mean"])
        smoothing = 10
        bin_stats["smooth_risk"] = (
            bin_stats["count"] * bin_stats["mean"] + smoothing * global_mean
        ) / (bin_stats["count"] + smoothing)
        bin_risk_map = bin_stats["smooth_risk"].to_dict()

    # Fallback prior 0.07 for unseen card BINs at inference time.
    df["card_bin_risk"] = df["card_bin"].map(bin_risk_map).fillna(0.07).astype(np.float32)

    df_indexed = df.set_index("timestamp")
    grouped = df_indexed.groupby("card_fingerprint")["amount_cents"]

    # Left-closed rolling intervals exclude the current transaction timestamp to prevent target leakage.
    for window, col_name in [
        ("5min", "card_tx_cnt_5m"),
        ("15min", "card_tx_cnt_15m"),
        ("60min", "card_tx_cnt_60m"),
    ]:
        df[col_name] = (
            grouped.rolling(window, closed="left")
            .count()
            .reset_index(level=0, drop=True)
            .fillna(0)
            .values.astype(np.float32)
        )

    df["card_sum_cents_60m"] = (
        grouped.rolling("60min", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .values.astype(np.float32)
    )

    feature_cols = [
        "amount_log",
        "is_round_amount",
        "amount_cents_mod_100",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "cvv_match_num",
        "is_3ds_passed_num",
        "risk_no_3ds_cvv_fail",
        "currency_code",
        "billing_country_code",
        "account_age_days",
        "account_age_days_log",
        "card_bin_risk",
        "card_tx_cnt_5m",
        "card_tx_cnt_15m",
        "card_tx_cnt_60m",
        "card_sum_cents_60m",
    ]

    return df, feature_cols, bin_risk_map


if __name__ == "__main__":
    raw_df = pd.read_csv("ml/preauth_transactions_enriched.csv", low_memory=False)
    processed_df, features, bin_map = build_features(raw_df, is_train=True)

    schema = {
        "features": features,
        "bin_risk_map": bin_map,
    }
    with open("model_artifacts/features_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
