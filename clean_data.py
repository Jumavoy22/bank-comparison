import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)

        df['trans_date'] = pd.to_datetime(df['trans_date'])
        df['total_sum'] = pd.to_numeric(df['total_sum'], errors='coerce').fillna(0)

        if 'p2p_flag' in df.columns:
            if df['p2p_flag'].dtype == 'object':
                df['p2p_flag'] = df['p2p_flag'].str.lower().isin(['true', '1', 't'])
            else:
                df['p2p_flag'] = df['p2p_flag'].astype(bool)
        else:
            df['p2p_flag'] = False

        df.dropna(how="all", inplace=True)
        df.drop_duplicates(inplace=True)

        crit_cols = [col for col in ["transaction_code", "trans_date", "total_sum", "bank_name"] if col in df.columns]
        df.dropna(subset=crit_cols, inplace=True)

        if 'bank_name' in df.columns:
            df["bank_name"] = df["bank_name"].str.strip().str.upper()
        if 'emitent_region' in df.columns:
            df["emitent_region"] = df["emitent_region"].str.strip().str.title()
        if 'gender' in df.columns:
            df["gender"] = df["gender"].str.strip().str.lower()

        if 'p2p_flag' in df.columns:
            df["p2p_flag"] = df["p2p_flag"].fillna(0).astype(bool)

        if 'total_sum' in df.columns and not df.empty:
            df = df[df["total_sum"] > 0]
            low, high = df["total_sum"].quantile([0.01, 0.99])
            df = df[(df["total_sum"] >= low) & (df["total_sum"] <= high)]

        return df

    except KeyError as e:
        raise KeyError(f"The file is missing a required column: {e}. Please ensure the file contains at least 'trans_date', 'total_sum', and 'bank_name'.")
    except Exception as e:
        raise IOError(f"Error reading or cleaning file: {e}")
