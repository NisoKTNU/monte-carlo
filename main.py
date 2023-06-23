import yfinance as yf
import pandas as pd
import math
from scipy.stats import norm
import numpy as np

ticker = "^SPX"
start_date = "2023-05-01"
end_date = "2023-06-01"
spx_data = yf.download(ticker, start=start_date, end=end_date)
simultations = 5
forward_period = 10


def daily_return():
    log_returns = spx_data['Close'].copy()
    previous_close = log_returns.shift(1)
    log_returns = log_returns / previous_close
    log_returns = log_returns.apply(lambda x: math.log(x))
    return log_returns


def drift(log_return):
    avg = log_return.sum()
    var = log_return.var()
    return avg - (var / 2)


def rand_val(log_return):
    sigma = log_return.std()
    return sigma * norm.ppf(np.random.rand())


def next_val(log_return, last_price):
    return last_price * math.e ** (drift(log_return) + rand_val(log_return))


def run_sim(log_return):
    sims = pd.DataFrame()
    for i in range(simultations):
        result = []
        for j in range(forward_period + 1):
            if j == 0:
                last_price = spx_data['Close'].iloc[-1]
            output = next_val(log_return, last_price)
            last_price = output
            result.append(output)
        sims[f'Sim {i+1}'] = result
    return sims


def main():
    log_return = daily_return()
    print(run_sim(log_return))


if __name__ == '__main__':
    main()
