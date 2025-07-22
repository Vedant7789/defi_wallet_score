from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def score_wallets(features_df: pd.DataFrame) -> pd.DataFrame:
    features_df['base_score'] = (
        features_df['borrow_repay_ratio'] * 400 +
        features_df['num_repay'] * 10 +
        features_df['total_repay'] * 0.01 -
        features_df['num_liquidations'] * 50
    )

    scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df['credit_score'] = scaler.fit_transform(features_df[['base_score']])
    return features_df[['wallet', 'credit_score']]

def save_scores(df, path="output/wallet_scores.csv"):
    df.to_csv(path, index=False)
