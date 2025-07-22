import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns for consistency
    df = df.rename(columns={
        'userWallet': 'user',
        'actionData': 'amount_data'
    })

    # Extract numeric amount from nested dict (actionData.amount)
    def extract_amount(row):
        try:
            return float(row['amount_data'].get('amount', 0))
        except:
            return 0

    df['amount'] = df.apply(extract_amount, axis=1)

    # Group by user (wallet address)
    grouped = df.groupby('user')
    features = []

    for wallet, group in grouped:
        actions = group['action'].value_counts().to_dict()
        total_transactions = len(group)
        total_deposit = group[group['action'] == 'deposit']['amount'].sum()
        total_borrow = group[group['action'] == 'borrow']['amount'].sum()
        total_repay = group[group['action'] == 'repay']['amount'].sum()
        liquidations = group[group['action'] == 'liquidationcall']

        features.append({
            'wallet': wallet,
            'num_txns': total_transactions,
            'num_deposit': actions.get('deposit', 0),
            'num_borrow': actions.get('borrow', 0),
            'num_repay': actions.get('repay', 0),
            'num_liquidations': len(liquidations),
            'borrow_repay_ratio': total_repay / total_borrow if total_borrow > 0 else 0,
            'total_deposit': total_deposit,
            'total_borrow': total_borrow,
            'total_repay': total_repay,
        })

    return pd.DataFrame(features)
