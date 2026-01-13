import numpy as np
# necessary for pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib as plt
from scipy.stats import norm
from typing import Optional
import math
# ---------------------------
# --------- LOGGING ---------
# ----- logging imports -----
import logging.config
import json
import pathlib
# instantiate logger
logger = logging.getLogger(__name__)

# --- LOGGER CONFIG ---
# Verbose dictionary-type config
#+for custom logger.
# Config file found in:
# /.config/logger/config.json
# (source: youtube.com/mCoding)
def setup_logging():
    config_file = pathlib.Path("config/logger/config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)

# initialize escape codes for
#+color-coding logs
red = "\033[31m"
green = "\033[32m"
yellow = "\033[33m"
blue = "\033[34m"
hl_red = "\033[41m"
hl_green = "\033[42m"
hl_yellow = "\033[43m"
hl_blue = "\033[44m"
reset = "\033[0m"
# ---------------------------
np.random.seed(7)

def process_spreadsheet(PATH_TO_CSV: pathlib.Path) -> pd.DataFrame:
    """Wrangle CSV trading data.

    Take a spreadsheet of trading data and 
    construct a pandas DataFrame with columns:
        date, close, and symbol.
    This function assumes OHLCV schema.

    Parameters
    ----------
    PATH_TO_CSV: 
        Path to CSV file with OHLCV trading data.
    
    Returns
    -------
    pd.Dataframe
        Pandas Dataframe with desired columns.
    """
    # read PATH_TO_CSV into a dataframe
    data = pd.read_csv(PATH_TO_CSV) 
    logger.debug(f"{yellow}CSV converted to Pandas Dataframe:{reset}\n{data.head()}")
    # ensure column names are lowercase
    data.columns = list(name.lower() for name in data.columns)
    data = data.rename(columns={'ts_event': 'date'})
    data["date"] = data["date"].astype(pd.DatetimeTZDtype(tz="America/New_York"))
    logger.debug(f"{yellow}Dataframe saved:{reset}\n{data.head()}")
    # select only date, close and symbol
    data = data[['date', 'close', 'symbol']]
    logger.info(f"\n{yellow}Records per symbol{reset}")
    logger.info(f"{data["symbol"].value_counts()}")
    logger.info(f"{hl_blue}Data extracted:{reset}\n{blue}{data.head()}{reset}")
    return data

def calculate_returns(close_data: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(close_data["symbol"].unique().tolist())
    new_columns = ["symbol", "date", "close", "returns", "log_close", "log_returns"]
    logger.debug(f"{yellow}Columns for new dataframes: {new_columns}{reset}")
    list_of_new_frames = []
    for ticker in tickers:
        temp_mask = close_data["symbol"] == ticker
        temp_frame = pd.DataFrame(close_data.loc[temp_mask], columns=new_columns)
        #logger.info(f"{yellow}Close data with symbol mask{reset}\n{close_data[temp_mask]}")
        temp_frame["returns"] = temp_frame.close.diff()
        temp_frame["log_close"] = np.log(temp_frame.close)
        temp_frame["log_returns"] = temp_frame.log_close.diff()
        temp_frame["log_returns_pct_change"] = temp_frame["log_returns"] * 100
        temp_frame = temp_frame.dropna()
        list_of_new_frames.append(temp_frame.copy())
        #logger.debug(f"{hl_blue}new frame created for {ticker}:{reset} \n{list_of_new_frames[-1]}")
    returns = pd.concat(list_of_new_frames)
    logger.debug(f"\n{hl_green}Hourly returns calculated:{reset}\n{green}{returns}{reset}")
    return returns

def calculate_metrics(return_data: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(return_data["symbol"].unique().tolist())
    means = []
    stds = []
    metrics = pd.DataFrame(index=tickers, columns=["log_mean", "log_std_dev"])
    logger.debug(f"{yellow}Initialized empty dataframe{reset} \n {metrics}")
    for ticker in tickers:
        means.append(float(return_data.loc[return_data["symbol"] == ticker, ["log_returns"]].mean()))
        stds.append(float(return_data.loc[return_data["symbol"] == ticker, ["log_returns"]].std()))
    means = np.array(means)
    stds = np.array(stds)
    metrics["log_mean"] = ((means + 1)**2016 - 1)  
    metrics["log_std_dev"] = (stds * np.sqrt(2016))

    logger.info(f"\n{hl_green}    Annualized Log-Return    {reset}\
                \n{hl_green} Means & Standard Deviations {reset}\
                \n{hl_green}            (in %)           {reset}\
                \n{green}{metrics}{reset}")
    return metrics

def correlation(return_data: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """
    This will construct a correlation matrix with given returns
    """
    logger.info(f"{hl_yellow}Calculating correlations for given returns{reset}\
                \n{yellow}{return_data}{reset}\
                \n{hl_yellow}With metrics{reset}\
                \n{yellow}{metrics}{reset}")
    tickers = sorted(return_data["symbol"].unique().tolist())
    list_of_metric_frames = []
    for ticker in tickers:
        temp_mask = return_data["symbol"] == ticker
        temp_frame = pd.DataFrame(return_data.loc[temp_mask], columns=["symbol", "date", "close", "return", "log_returns"])
        temp_frame["normalized_returns"] = temp_frame.close - metrics.loc[ticker][0]
        list_of_metric_frames.append(temp_frame.copy())
    data = pd.concat(list_of_metric_frames)
    logger.info(f"{hl_blue} Returns Normalized {reset}\
                \n {data[["symbol", "normalized_returns"]]}")
    #counts = return_data["symbol"].value_counts()
    #counts = counts.values.reshape(-1, 1)
    #avg_observations = (counts + counts.T) / 2
    #prefactors = 1 / (avg_observations - 1)
    #logger.info(f"{hl_yellow}Prefactors:{reset} \
                #\n {prefactors}")
    # =====================
    # 1. Pivot so symbols are columns and dates are rows
    # We use 'log_returns' as it's the standard for financial covariance
    pivoted_returns = data.pivot(index='date', columns='symbol', values='log_returns')

    # 2. Compute the pairwise covariance matrix
    # This method automatically handles missing values (NaNs)
    #covariance_matrix = pivoted_returns.cov() * 2016
    correlation_matrix = pivoted_returns.corr()

    #logger.info(covariance_matrix)
    logger.info(f"{hl_green} Correlation Matrix {reset} \
                \n{green}{correlation_matrix}{reset}")
    logger.info(f"{hl_green}Eigenvalues of the correlation matrix:{reset}\
                 \n{green}{np.linalg.eigvals(correlation_matrix)}{reset}")
    # =====================
    logger.info(f"{hl_green}Cholesky Decomposition{reset}\
                \n {green}{np.linalg.cholesky(correlation_matrix)}{reset}")
    return correlation_matrix

def gbm_sim_antithetic(
    s0: float, drift: float, diffusion: float, n: int, t: float
) -> np.ndarray:
    """
    Returns n simulated paths (assuming input n is even) where the second half
    of the simulations are the antithetic of the first half
    """
    mean = drift * t
    std_dev = diffusion * np.sqrt(t)
    std_normal = norm(loc=0, scale=1)
    n_use = int(n / 2)
    es = std_normal.rvs(n_use)
    xis = np.concatenate([mean + std_dev * es, mean - std_dev * es])
    return s0 * np.exp(xis)

def gbm_sim(
        s0: float, 
        drift: float, 
        diffusion: float, 
        n: int, 
        t: float,
        rng: Optional[np.random.RandomState | np.random.Generator] = None,
        ) -> np.array:
    """
        This function generates a geometric brownian motion path with
            the below given parameters
            
        s0: initial price
        drift: gbm drift
        diffusion: gbm diffusion
        n: # of timesteps
        t: term
        rng: random number generator
    """
    # time step is given by t: time horizon
    #+divided by the number of calculated prices n
    dt = t / n
    # normally distributed factors given by black scholes equation
    xis = norm.rvs(loc=drift * dt, scale=diffusion * np.sqrt(dt), size = n)
    return s0 * np.exp(np.cumsum(xis))

def generate_matrix(
        price_zero: float, 
        m: int, 
        n: int, 
        T: float,
        r: float,
        v: float,
        q: Optional[float] = 0,
        ) -> np.array:
    """
        This function will generate a matrix of simulated stock prices
            based on a Geometric Brownian Motion.

        price_zero: initial stock price at time 0
        m: # of paths
        n: # of timesteps
        T: term of the projection
        r: continuous risk free rate
        v: volatility
        q: continuous dividend yield (default = 0)
        drift: drift (default = None)
    """
    # initialize an M x N + 1 matrix with zeros
    price_paths = np.zeros((m, n + 1))
    # set the initial price to [price_zero] in every
    #+row.
    price_paths[:, 0] = price_zero
    # Drift and diffusion parameters in log price process
    #def calc_drift(value: Optional[float] = 0.085) -> float:
    #    drift = r - q - 0.5 * v**2
    #    return value
    #path_drift = calc_drift()
    path_drift = r - q - 0.5 * v**2
    path_diffusion = v
    dt = T / n
    xis = norm.rvs(loc=path_drift * dt, scale=path_diffusion * np.sqrt(dt), size = n)

    for i in range(m):
        price_paths[i, 1:] = gbm_sim(price_zero, path_drift, path_diffusion, n, T)
    return price_paths

def main():
    setup_logging()
    logger.info(f"{green}Logging initialized.{reset}")
    file = pathlib.Path("data/databento/xnas-itch-20180501-20260107.ohlcv-1h.csv")
    df = process_spreadsheet(file)
    returns = calculate_returns(df)
    matrix = calculate_metrics(returns)
    correlations = correlation(returns, matrix)
if __name__ == "__main__":
    main()
