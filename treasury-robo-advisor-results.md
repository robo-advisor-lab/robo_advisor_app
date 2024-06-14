---
description: v 0.1 results
---

# Treasury Robo Advisor Results

### Configuration

num\_runs = 10 seeds = \[5, 10, 15, 20, 40, 100, 200, 300, 500, 800]&#x20;

eth\_bound = 0.2&#x20;

rebalancing\_frequency = 15&#x20;

start\_date = panama\_dao\_start\_date - 03-22-2023&#x20;

end\_date = panama\_dao\_end\_date - 05-15-2024 &#x20;

all\_assets = \['COMP', 'CWBTC', 'DPI', 'DYDX', 'ETH', 'FXS', 'GFI', 'MATIC', 'MKR', 'PAXG', 'RETH', 'RPL', 'SOL', 'UNI', 'USDC', 'WBTC', 'WETH', 'WSTETH']

### Results

Sortino Ratios:

* Historical Sortino Ratio: 1.81
* Average RL Sortino Ratio: 3.45&#x20;
* MVO Sortino Ratio: 1.51

Cumulative Returns:

* Historical Cumulative Return: 0.62
* Average RL Cumulative Return: 1.20&#x20;
* MVO Cumulative Return: 0.73

CAPM Results:

Benchmark - [Sirloin Index](https://dune.com/queries/2233092)

Cumulative Risk Premium: 0.35

Risk free = 0.0524

* Historical Beta: 1.74
* Historical CAGR: 0.63
* Avg RL Beta: 2.7
* Avg RL CAGR: 1.45
* MVO Beta: 2.18
* MVO CAGR: 1.05



<figure><img src=".gitbook/assets/newplot (30).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (31).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (32).png" alt=""><figcaption></figcaption></figure>

### Now lets look at one run in particular

Seed: 100

Initial amount: $100.00

Rebalancing Frequency: 15 days

ETH Minimum Bound: 20.00%

Selected Assets: \['COMP', 'CWBTC', 'DPI', 'DYDX', 'ETH', 'FXS', 'GFI', 'MATIC', 'MKR', 'PAXG', 'RETH', 'RPL', 'SOL', 'UNI', 'USDC', 'WBTC', 'WETH', 'WSTETH']



RL Adjusted Return

$234

RL Cumulative Return

133.95%

RL Sortino

3.86

MVO Adjusted Return

$174

MVO Cumulative Return

74.00%

MVO Sortino

1.51

Historical Adjusted Return

$162

Historical Cumulative Return

61.80%

Historical Sortino

1.81

\


<figure><img src=".gitbook/assets/newplot (23).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (24).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (26).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (27).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (28).png" alt=""><figcaption></figcaption></figure>

RL Beta

3.38

RL CAGR

169.92%

MVO Beta

2.18

MVO CAGR

105.33%

Historical Beta

1.74

Historical CAGR

63.39%

<figure><img src=".gitbook/assets/newplot (29) (1).png" alt=""><figcaption></figcaption></figure>



