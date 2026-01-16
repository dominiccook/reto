# Inputs: N assets
#          -expected return
#          -standard deviation
#         Correlation matrix
#          -N x N 
#         Assumptions
#          -spending level, inflation rate, tax rate, standard deduction, etc.
#
# Simulation parameters:  Time horizon in years
#                         Number of sample market paths
#                         Number of sample portfolio weights, non-negative and sum to 1
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Approach: 1) Generate future correlated market returns across assets in portfolio
#               a) ensure correlation matrix is positive semi-definite
#               b) construct covariance matrix
#               c) perform Cholesky decomposition
#               d) for each year in the time horizon, create a matrix of market returns. dimensions will be number of assets x number of sample market paths
#               e) inflate spending
#           
#           2) Generate set of portfolio weights and run each portfolio through the future markets/spending
#               a) [weights x market return along each path - spending]
#               b) rebalance remaining funds each year back to original portfolio weight and move to next year generated set of correlated market returns
#               c) floor any paths at 0 and maintain that path at 0 for all future time steps
#
#           3) After full set of portolios is run through each year, count number of market paths that don't drop to 0 for 
#              each portfolio weighting and divide by total number of simulated paths to get % chance of success
#
#           4) Return the single portfolio weighting with the highest expected success rate
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# To-dos: expected return, standard deviation, correlation matrix calculations
#         understand output sensitivity when trimming negative eigenvalues to ensure positive semi-definiteness  
#         figure out more efficient method to run a given portfolio of weights across market returns and spending, currently done in a slow recursive loop
#         generate random correlated market returns from distributions other than normal, Metropolis-Hastings/Gibbs sampling
#         investigate hedging strategies to raise portfolio success rate - new investment assets, VIX, long puts, collars funded with OTM calls 
#         research stability of initial optimal portfolio through actual realized years
#         if flexibility in spending is allowed, switch from fixed $ distribution to % remaining asset distribution and quantify measures of portfolio shortfalls
#         as complexity increases/dimensions of solution options expands may need to incorporate simulated annealing/filled function method/evolutionary algos/etc.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from datetime import datetime

import allocate_net_worth as anw

from pathlib import Path
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
#def setup_logging():
#    config_file = pathlib.Path("config/logger/config.json")
#    with open(config_file) as f_in:
#        config = json.load(f_in)
#    logging.config.dictConfig(config)

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

#time_horizon = 50 # Retirement horizon in years
#n_samples = 1000  # Number of paths
#spending_necessary = 40000 # annual
#spending_unnecessary = 0 
#spending_total = 40000 
#inflation_rate = .03
#tax_rate = .01 # not currently used
#standard_deduction = 30000 # not currently used
#current_year = datetime.now().year

def success_rate(starting_balance: int=1000000,
                 time_horizon: int=50,
                 n_samples: int=1000,
                 necessary_drawdown: float=0.04,
                 unnecessary_drawdown: float=0,
                 inflation_rate: float=0.03,
                 tax_rate: float=0.01,
                 standard_deduction: int=30000,               
                 current_year: datetime=datetime.now().year,                
                 returns_std_devs: Path="data/returns_std_devs.csv",
                 index_correlations: Path="data/index_correlations.csv",
                 balances_and_target_allocations: Path="data/balances_and_target_allocations",
                 hedge_allocation: float=0,
                 put_floor: float=-0.10,
                 put_cost_rate: float=0.02,
                 config_file: Path="config/logger/config.json",
                 log: bool=False
                 ):
    """Modern Portfolio Success Rate
    
        Parameters
        ----------
            starting_balance:
                Default 1 million

            time_horizon:
                Retirement horizon in years

            n_samples:
                Number of paths generated

            necessary_drawdown:
                Annual necessary spending as a percentage of initial balance.

            unnecessary_drawdown:
                Annual unnecessary spending.

            spending_total:

            inflation_rate:

            tax_rate:           #not currently used

            standard_deduction: #not currently used

            current_year:

            returns_std_devs:

            index_correlations:

            balances_and_target_allocations:

            hedge_allocation:

            put_floor:

            put_cost_rate:


    """
    if (log):
        with open(config_file) as f_in:
            config = json.load(f_in)
        logging.config.dictConfig(config)
        logger.info(f"{green}Logging initialized.{reset}")

    # Load starting balances, target allocations, returns/std devs, correlations from CSV files into DataFrames
    file_path = returns_std_devs
    df_Returns_Std_Deviations = pd.read_csv(file_path)

    file_path2 = index_correlations
    df_Index_Correlations = pd.read_csv(file_path2)

    file_path3 = balances_and_target_allocations 
    df_Balances_and_Target_Allocations = pd.read_csv(file_path3)

    # Beg sum of all assets
    Beg_Net_Worth = df_Balances_and_Target_Allocations['Balance'].sum()

    # Format manipulation of input files to perform mathematical operations
    v_mu = np.array(df_Returns_Std_Deviations['Mean']).astype(float)
    v_sigma = np.array(df_Returns_Std_Deviations['Standard_Deviation']).astype(float)

    if starting_balance != Beg_Net_Worth:
        Beg_Net_Worth = starting_balance
        adjusted_balances = anw.allocate(net_worth=Beg_Net_Worth ,file=balances_and_target_allocations)
        df_Balances_and_Target_Allocations['Balance'] = adjusted_balances

    # Utility function to force correlation matrix to be positive semi definite
    def make_positive_semi_definite(matrix):
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues to a small positive value
        eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
        
        # Reconstruct the matrix
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return psd_matrix

    # Convert correlation matrix to covariance matrix
    correlation_columns = np.array(df_Index_Correlations[['voo', 'vti', 'vwo', 'vtv', 'vb', 'bsv', 'vo', 'biv', 'vv', 'blv']]).astype(float)
    positive_definite_matrix = make_positive_semi_definite(correlation_columns)
    cov_matrix = np.diag(v_sigma) @ positive_definite_matrix @ np.diag(v_sigma)

    # Perform Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)

    # Initialize spending matrix over time horizon
    spending_necessary = Beg_Net_Worth * necessary_drawdown
    spending_unnecessary = Beg_Net_Worth * unnecessary_drawdown
    spending_matrix = pd.DataFrame(index=range(time_horizon), columns=['Necessary', 'Unnecessary', 'Total'])
    beginning_drawdown = necessary_drawdown + unnecessary_drawdown

    # Initial market cube returns 
    Market_Cube_Returns = {}
    Market_Cube_Returns_plus1 = {}

    #hedge_allocation = 0  # Between 0 and 1, with 0 meaning portfolio is not hedged at all and 1 meaning hedge is applied across the whole portfolio
    #put_floor = -0.10       # Put option caps losses at 10% per year
    #put_cost_rate = 0.02        # 2% annual cost for the put option

    def apply_put_hedge_with_cost(returns, hedge_allocation, put_floor, put_cost_rate):
        # returns: shape (n_samples, n_assets)
        # For the hedged portion, cap losses at put_floor
        hedged_returns = np.where(returns < put_floor, put_floor, returns)
        # Subtract put cost from hedged portion
        hedged_returns = hedged_returns - put_cost_rate
        # Combine hedged and unhedged portions
        total_returns = hedge_allocation * hedged_returns + (1 - hedge_allocation) * returns
        return total_returns

    # Generate uncorrelated normal random variables
    for i in range(time_horizon):
        
        # Generate uncorrelated random variables
        uncorrelated = np.random.normal(size=(n_samples, len(v_mu)))

        # Transform to correlated random variables
        correlated = uncorrelated @ L.T + v_mu
        # Apply put hedge to returns
        hedged_correlated = apply_put_hedge_with_cost(correlated, hedge_allocation, put_floor, put_cost_rate)
        correlated_plus1 = hedged_correlated + 1
        
        # Store in dictionary
        Market_Cube_Returns[f'Market_Returns_{int(current_year)+i}'] = hedged_correlated.T
        Market_Cube_Returns_plus1[f'Market_Returns_{int(current_year)+i}'] = correlated_plus1.T
        
        text = int(current_year) + i
        
        # Populating future spending matrix
        spending_total = spending_necessary + spending_unnecessary
        spending_matrix.loc[i] = [spending_necessary, spending_unnecessary, spending_total]
        
        # Inflate spending for next time period
        spending_necessary = spending_necessary * (1 + inflation_rate)
        spending_unnecessary = spending_unnecessary * (1 + inflation_rate)
        spending_total = spending_necessary + spending_unnecessary

    # Calculate mean returns and correlation matrix of generated market
    mean_returns = np.mean(correlated, axis = 0)
    corr_matrix_check = np.corrcoef(correlated.T)
    num_assets = len(mean_returns)

    # Simulate random portfolios
    num_portfolios = 1000
    results = np.zeros((4, num_portfolios))
    weights_record = []
    End_of_Period_Balances_record = []
    Beg_of_Period_Rebalanced_record = []

    # Create random portfolio weights and run each one through the generated market returns
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        #weights = np.array([.3, .1, .2, .1, .05, .05, .05, .05, .05, .05]) # override for desired allocation
        
        # Portfolio performance
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_stddev
        
        # Record every portfolio return, standard deviation, and sharpe ratio
        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio
        
        for j in range(time_horizon):
            # Run each portfolio allocation i through each year j
            if j == 0:
                Beg_of_Period_Net_Worth =  max(np.sum(df_Balances_and_Target_Allocations.iloc[:,1]) - spending_matrix.loc[j,'Total'], 0)
                Beg_of_Period_Rebalanced = weights * Beg_of_Period_Net_Worth
                Beg_of_Period_Rebalanced_record.append(Beg_of_Period_Rebalanced)
                
                End_of_Period_Balances = Beg_of_Period_Rebalanced * Market_Cube_Returns_plus1.get(f'Market_Returns_{current_year+j}').T  
                End_of_Period_Balances_record.append(End_of_Period_Balances)
                End_of_Period_Net_Worth = np.sum(End_of_Period_Balances, axis=1)                                                         
                End_of_Period_Avg_Net_Worth = np.average(End_of_Period_Balances, axis = 0)
            else:
                Beg_of_Period_Net_Worth = End_of_Period_Net_Worth - spending_matrix.loc[j,'Total']
                Beg_of_Period_Net_Worth[Beg_of_Period_Net_Worth < 0] = 0 # Negative value portfolios floored at 0 balance
                Beg_of_Period_Rebalanced = np.outer(Beg_of_Period_Net_Worth, weights)
                Beg_of_Period_Rebalanced_record = list(zip(Beg_of_Period_Rebalanced_record, Beg_of_Period_Rebalanced))
                
                End_of_Period_Balances = Beg_of_Period_Rebalanced * Market_Cube_Returns_plus1.get(f'Market_Returns_{current_year+j}').T
                End_of_Period_Balances_record = list(zip(End_of_Period_Balances_record, End_of_Period_Balances))
                End_of_Period_Net_Worth = np.sum(End_of_Period_Balances, axis=1)
                End_of_Period_Avg_Net_Worth = np.average(End_of_Period_Balances, axis = 0)

        # Record % chance of sucess for each portfolio based on path level results 
        results[3, i] = np.count_nonzero(End_of_Period_Balances.T[-1])/n_samples

    # Extract results
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_return = results[0, max_sharpe_idx]
    max_sharpe_stddev = results[1, max_sharpe_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]

    max_port_success_idx = np.argmax(results[3]) # Index location of max success rate in results table
    max_port_success_return = results[0, max_port_success_idx]
    max_port_success_stddev = results[1, max_port_success_idx]
    max_port_success_weights = weights_record[max_port_success_idx]

    # Plot Efficient Frontier
    plt.scatter(results[1, :], results[0, :], c=results[3, :], cmap='viridis', marker='o')
    plt.colorbar(label='Portfolio Success') 
    plt.scatter(max_port_success_stddev, max_port_success_return, color='red', marker='*', s=200, label='Max Port Success Rate')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

    # Display optimal weights
    print(f"{green} Retiring with initial balance of {Beg_Net_Worth}\
          \n and an annual withdrawal of {beginning_drawdown}\
          \n for {time_horizon} years, yields: {reset}")
    print("Optimal Weights for Portfolio Longevity:")
    for i, stock in enumerate(df_Index_Correlations.iloc[:,1:]):
        print(f"{stock}: {max_port_success_weights[i]:.2%}")
    print("Portfolio Success Rate:", results[3,np.argmax(results[3])])
    print("Portfolio Expected Return:", results[0,np.argmax(results[3])])
    print("Portfolio Standard Deviation:", results[1,np.argmax(results[3])])

if __name__ == "__main__":
    success_rate()