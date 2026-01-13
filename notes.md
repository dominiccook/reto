monday evening zoom meeting check-in saturday presentations

Derek presentation Increasing the sustainable withdrawal rate of risky
portfolios

interest rate, commodities, FX (as well as credit) at US Bank Q: do you know
korbyn good?

Personal project for derek

Problem last 1 or 2 generatons, shift in approach to retirement defined benefit
plans (pay in money, guaranteed money upon retirement) defined contribution
(401k etc) (variable money upon retirement)

How do you use your retirement reserves 

reduce dimensionality E: mortality

Q: what simplifuying assumptopns

Goal: hedge downside risk in early years of retirement

60yo retirement

Time considerations related Tail event risks (covid, 2008, 87 crash)

good: larger portfolio and long runway

average return over 10years sequence of returns fixed-dollar distribution

Assumptions & data: use bloomberg terminal, grab historical data for a desired
set of index funds and calculate returns, std dev, and correlation matrix

Q: how do you construct a correlation matrix?

simulate future markets

starting balance is $1 million and beginning spending is $40k/yr, no taxes
inflation at 3%, time horizon of 50years numbers and rates are on an annual
basis (more important is the withdrawal rate)

E: its better to have a longer runway

q: can i find historical inflation curve to generate a future one?


## Generate future markets future correlated market returns across assets in portfolio
- ensure correlation matrix is positive semi-definite
- contruct covariance matrix
- perform cholesky decomposition
- for each year in the time horizon, create a matrix of market returns.
  Ultimately this provides a cube with dimensions [# assets, # paths, horizon]

  Market return cube

## Incorporate spending, test allocations, and count the number of paths that
don't drop below $0

Generate set of portgolio weights and run each portfolio thorugh the future
markets /spending


example:
- buy puts to hedge downside
- write calls to generate income
- time based asset allocation

does it make sense to change asset allocation as you age?


Questions:

Q: Is there any consideration for inflation heteroskedasticity?
Q: which index funds are desired (which ones will we use data from)?

correltion between index fund returns and inflation

what can we do to protect ourselves given a longer runway
5 or 10 index funds
generate future markets
LCD decomposition
Chelesky decomposition
generated correlated market returns for index funds
    matrix with correlated returns from different indexes

    pairwise correlations for all index fund combinations
    1000 path simulation
    averagre the 1000 path
    8% return 15% std dev should match S&P
    Future market correlation matrix
    - Cholesky



Dimensions
starting balance

inflation

assets/index funds (expected return, std dev)
correlation matrix (nxn n = number of funds)





























VOO	(Vanguard S&P 500 ETF)

BSV	(Vanguard Short-Term Bond ETF)
BIV	(Vanguard Intermediate-Term Bond ETF)
BLV	(Vanguard Long-Term Bond ETF)

VB	(Vanguard Small-Cap ETF)
VO	(Vanguard Mid-Cap ETF)
VV	(Vanguard Large-Cap ETF)

VTV	(Vanguard Value ETF)
VTI	(Vanguard Total Stock Market ETF)
VWO	(Vanguard FTSE Emerging Markets ETF)
