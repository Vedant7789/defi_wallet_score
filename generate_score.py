import json
import pandas as pd
import os

from feature_engineering import extract_features
from utils import score_wallets, save_scores

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Step 1: Load transaction data
print("🔄 Loading transaction data...")
with open("data/user_transactions.json") as f:
    transactions = json.load(f)

# Step 2: Convert to DataFrame
print("📄 Converting to DataFrame...")
df = pd.DataFrame(transactions)

# Optional validation
if 'action' not in df.columns or 'userWallet' not in df.columns:
    raise ValueError("Missing required columns like 'action' or 'userWallet'")

# Step 3: Feature Engineering
print("🧠 Extracting features...")
features_df = extract_features(df)

# Step 4: Scoring
print("📊 Scoring wallets...")
scored_df = score_wallets(features_df)

# Step 5: Save Results
print("💾 Saving results to output/wallet_scores.csv...")
save_scores(scored_df)

print(f"✅ Done! Scored {len(scored_df)} wallets.")
