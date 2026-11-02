# wheel_tna.py

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================

TICKER = "TNA"
START = "2010-01-01"
END = None

INITIAL_CAPITAL = 100000

# Option assumptions
DTE = 38                     # middle of 30-45 DTE
TARGET_DELTA = 0.30
RF_RATE = 0.04
IV_MULTIPLIER = 1.15         # IV > RV

# =========================
# BLACK-SCHOLES HELPERS
# =========================

def put_delta(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    return norm.cdf(d1) - 1


def call_delta(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    return norm.cdf(d1)


def option_price_put(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


def option_price_call(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def find_strike_for_delta(
        S,
        sigma,
        target_delta,
        T,
        r,
        put=True):

    strikes = np.arange(
        S * 0.50,
        S * 1.50,
        max(0.25, S * 0.0025)
    )

    best_strike = None
    best_diff = 999

    for K in strikes:

        if put:
            d = abs(put_delta(
                S,
                K,
                T,
                r,
                sigma
            ))
        else:
            d = call_delta(
                S,
                K,
                T,
                r,
                sigma
            )

        diff = abs(d - target_delta)

        if diff < best_diff:
            best_diff = diff
            best_strike = K

    return best_strike


# =========================
# DATA
# =========================

df = yf.download(
    TICKER,
    start=START,
    end=END,
    auto_adjust=True
)

df["RET"] = df["Close"].pct_change()

# 30-day realized vol
df["RV30"] = (
    df["RET"]
    .rolling(30)
    .std()
    * np.sqrt(252)
)

df.dropna(inplace=True)

# =========================
# WHEEL BACKTEST
# =========================

cash = INITIAL_CAPITAL
shares = 0

trade_log = []

equity_curve = []

position = "CASH"

i = 0

while i < len(df) - DTE:

    current_date = df.index[i]
    current_price = float(df["Close"].iloc[i])

    sigma = max(
        0.20,
        float(df["RV30"].iloc[i]) * IV_MULTIPLIER
    )

    T = DTE / 365

    expiration_idx = i + DTE

    expiration_price = float(
        df["Close"].iloc[expiration_idx]
    )

    # ==================================================
    # CASH SECURED PUT
    # ==================================================

    if shares == 0:

        strike = find_strike_for_delta(
            current_price,
            sigma,
            TARGET_DELTA,
            T,
            RF_RATE,
            put=True
        )

        premium = option_price_put(
            current_price,
            strike,
            T,
            RF_RATE,
            sigma
        )

        cash += premium * 100

        assigned = expiration_price < strike

        trade_log.append({
            "Date": current_date,
            "Type": "PUT",
            "Strike": strike,
            "Premium": premium,
            "Assigned": assigned
        })

        if assigned:

            shares = 100
            cash -= strike * 100

        i += DTE

    # ==================================================
    # COVERED CALL
    # ==================================================

    else:

        strike = find_strike_for_delta(
            current_price,
            sigma,
            TARGET_DELTA,
            T,
            RF_RATE,
            put=False
        )

        premium = option_price_call(
            current_price,
            strike,
            T,
            RF_RATE,
            sigma
        )

        cash += premium * 100

        called_away = expiration_price > strike

        trade_log.append({
            "Date": current_date,
            "Type": "CALL",
            "Strike": strike,
            "Premium": premium,
            "CalledAway": called_away
        })

        if called_away:

            cash += strike * 100
            shares = 0

        i += DTE

    current_share_value = (
        shares * expiration_price
    )

    total_equity = (
        cash
        + current_share_value
    )

    equity_curve.append({
        "Date": df.index[min(expiration_idx, len(df)-1)],
        "Equity": total_equity
    })

# =========================
# RESULTS
# =========================

eq = pd.DataFrame(equity_curve)

eq["Return"] = eq["Equity"].pct_change()

years = (
    (eq["Date"].iloc[-1]
     - eq["Date"].iloc[0]).days
    / 365.25
)

ending_value = eq["Equity"].iloc[-1]

cagr = (
    ending_value
    / INITIAL_CAPITAL
) ** (1 / years) - 1

rolling_max = eq["Equity"].cummax()

drawdown = (
    eq["Equity"]
    / rolling_max
    - 1
)

max_dd = drawdown.min()

sharpe = (
    eq["Return"].mean()
    / eq["Return"].std()
    * np.sqrt(252 / DTE)
)

# =========================
# BUY & HOLD COMPARISON
# =========================

initial_price = float(df["Close"].iloc[0])

bh_shares = INITIAL_CAPITAL / initial_price

bh_end = bh_shares * float(df["Close"].iloc[-1])

bh_cagr = (
    bh_end
    / INITIAL_CAPITAL
) ** (1 / years) - 1

# =========================
# OUTPUT
# =========================

print("\n===== TNA WHEEL =====")
print(f"Ending Value: ${ending_value:,.2f}")
print(f"CAGR: {cagr:.2%}")
print(f"Max DD: {max_dd:.2%}")
print(f"Sharpe: {sharpe:.2f}")

print("\n===== BUY HOLD =====")
print(f"Ending Value: ${bh_end:,.2f}")
print(f"CAGR: {bh_cagr:.2%}")

print("\nTrades:")
print(len(trade_log))

# =========================
# CHART
# =========================

plt.figure(figsize=(10,5))
plt.plot(
    eq["Date"],
    eq["Equity"],
    label="Wheel"
)

plt.title(
    "TNA Wheel Strategy Equity Curve"
)

plt.legend()
plt.grid(True)

plt.show()
