tickers = ('SPY',
           'AAPL',
           'AMZN',
           'JPM',
           'WMT',
           'TSLA',
           'HD',
           'JNJ',
           'QCOM',
           'MSFT',
           'GS',
           'VZ')


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# Calculates mean, standard deviation, and the covariance matrix for each stock
def stats(x):
    m, n = x.shape
    mu = (1/m)*np.ones(m).dot(x)
    cov = (1/(m-1))*(x - mu).T.dot(x - mu)
    sd = np.sqrt(np.diag(cov))
    return sd, mu, cov

# Calculates efficient frontier curve
def efficient_frontier(mu, cov, n=25):

    # Optimizes at each inputted return to return min variance portfolio
    def optimize(rate):
        # Objective function
        def obj(x):
            return -(x.T.dot(mu))/(x.T.dot(cov.dot(x)))
        
        # Constraint to equal target return
        def cons(x):
            return x.T.dot(mu) - rate

        # Constraint to make all weights sum equal one
        def zeroz(x):
            return np.sum(x) - 1

        # Initialize weights list
        x = [0.1 for i in mu]

        # Write constraints
        constraints = [{'type':'eq','fun':cons},
                       {'type':'eq','fun':zeroz}]

        # Optimize function
        res = minimize(obj, x, method='SLSQP', bounds=None, constraints=constraints)
        return res.x

    # Pick range of rates to input
    r0, r1 = np.min(mu), np.max(mu)
    dR = (r1 - r0)/(n - 1)
    r = [r0 + i*dR for i in range(n)]

    # Build the curve
    kx, ky = [], []
    for rate in r:
        w = optimize(rate)
        risk = np.sqrt(w.T.dot(cov.dot(w)))
        retz = w.T.dot(mu)
        kx.append(float(risk))
        ky.append(float(retz))

    return kx, ky

# Calculates min-variance portfolio
def minvar(cov):
    cv = (2*cov).tolist()
    for i in range(len(cv)):
        cv[i].append(1)
    cv.append([1 for i in cov] + [0])
    z = np.array(cv)
    b = np.array([[0] for i in cov] + [[1]])
    return np.linalg.inv(z).dot(b)[:-1]


# Calculates max-sharpe portfolio
def max_sharpe(mu, cov):
    def obj(x):
        return -(x.T.dot(mu))/(x.T.dot(cov.dot(x)))
    def cons(x):
        return np.sum(x) - 1
    const = [{'type':'eq','fun':cons}]
    x = [0.1 for i in mu]
    res = minimize(obj, x, method='SLSQP', bounds=None, constraints=const)
    return res.x

# Calculates the capital allocation line
def CAL(x0, y0):
    # Preset weights
    w = (0.3, 1, 2)
    ux, uy = [], []
    # Calculation of risk/return
    for weight in w:
        ux.append(weight*x0)
        uy.append(weight*y0 + (1-weight)*-y0)
    return ux, uy



# Import the data csv files in a pandas frame
data = {tick:pd.read_csv(f'{tick}.csv') for tick in tickers}

# Extract adj close prices
close = np.array([data[tick]['adjClose'].values.tolist() for tick in tickers]).T

# Calculate the rate of return of each stock
ror = close[:-1]/close[1:] - 1

# Fetch stdev, mean, and covariance
sd, mu, cov = stats(ror)

# Initialize a figure and a plot
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
ax.set_xlabel("Risk (Standard Deviation)")
ax.set_ylabel("Average Daily Return")

# Plot each stocks risk/return and label it by ticker
for t, x, y in zip(tickers, sd, mu):
    ax.scatter(x, y, color='blue', s=10)
    ax.scatter(x, y, color='cyan', s=3)
    ax.annotate(t, xy=(x, y))

# Plot EF
qx, qy = efficient_frontier(mu, cov)
ax.plot(qx, qy, color='red', linewidth=1.5)

# Plot Min-Var Portfolio
wm = minvar(cov)
risk_x = np.sqrt(wm.T.dot(cov.dot(wm))[0][0])
retz_y = wm.T.dot(mu)[0]
ax.scatter(risk_x, retz_y, color='black', s=40)
ax.annotate('Min-Var Port.', xy=(risk_x, retz_y), rotation=45)

# Plot Max-Sharpe Portfolio
ws = max_sharpe(mu, cov)
risk_k = np.sqrt(ws.T.dot(cov.dot(ws)))
retz_k = ws.T.dot(mu)
ax.scatter(risk_k, retz_k, color='black', s=40)
ax.annotate('Max-Sharpe Port.', xy=(risk_k, retz_k), rotation=28)

# Plot Capital Allocation Line
tx, ty = CAL(risk_k, retz_k)
ax.plot(tx, ty, color='black')

# Show plot
plt.show()

