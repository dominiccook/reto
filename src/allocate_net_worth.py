import numpy as np
import pandas as pd
from pathlib import Path

def allocate(file: Path="data/balances_and_target_allocations.csv", 
             net_worth: int=1000000,
             ) -> np.ndarray:
    df = pd.read_csv(file)
    default_balances = df.Balance.to_numpy()
    default_net_worth = np.sum(df.Balance)
    default_weights = default_balances / default_net_worth
    new_balances = default_weights * net_worth
    return new_balances

if __name__ == "__main__":
    allocate()