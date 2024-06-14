---
description: v 0.1 results
---

# Vault Robo Advisor Results

bounds = { 'BTC Vault\_collateral\_usd': (0.1, 0.3), 'ETH Vault\_collateral\_usd': (0.1, 0.5), 'stETH Vault\_collateral\_usd': (0.2, 0.3), 'Stablecoin Vault\_collateral\_usd': (0.0, 0.05), 'Altcoin Vault\_collateral\_usd': (0.0, 0.05), 'LP Vault\_collateral\_usd': (0.05, 0.05), 'RWA Vault\_collateral\_usd': (0.05, 0.05), 'PSM Vault\_collateral\_usd': (0.05, 0.2) }

initial strategy =

{ "BTC Vault\_collateral\_usd": 0.1, "ETH Vault\_collateral\_usd": 0.1999999999999995, "stETH Vault\_collateral\_usd": 0.3, "Stablecoin Vault\_collateral\_usd": 0.049999999999999496, "Altcoin Vault\_collateral\_usd": 0.049999999999999045, "LP Vault\_collateral\_usd": 0.05, "RWA Vault\_collateral\_usd": 0.05, "PSM Vault\_collateral\_usd": 0.2 }

vault\_action\_ranges = { 'stETH Vault\_dai\_ceiling': \[-0.5, 0.5], 'ETH Vault\_dai\_ceiling': \[-1, 0.5], 'BTC Vault\_dai\_ceiling': \[-1, 0.5], 'Altcoin Vault\_dai\_ceiling': \[-0.15,0.15], 'Stablecoin Vault\_dai\_ceiling': \[-0.1, 0.1], 'LP Vault\_dai\_ceiling': \[-0.1,0.1], 'RWA Vault\_dai\_ceiling': \[0, 0], 'PSM Vault\_dai\_ceiling': \[-1, 0.5] }

start\_date = '2022-05-20'&#x20;

end\_date = '2024-03-20'

num\_runs = 10 seeds = \[5, 10, 15, 20, 40, 100, 200, 300, 500, 800]

initial\_strategy\_period=1

### RL Average Scores

* avg rl cum return = -0.19
* avg rl tvl = 10,629,002,211.
* avg rl sortino = -0.22

### MVO Scores

* mvo cum return = -0.022
* mvo tvl = $`7195864283.12639`
* mvo sortino = -0.19

### Historical Scores

* Cumulative Return: -0.34
* TVL: 7,342,533,784.
* Portfolio Sortino Ratio as of 3/20/2024: -0.55

<figure><img src=".gitbook/assets/rl_vault_cum_run.png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_vault_cum_run_tvl.png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_vault_sortino_run.png" alt=""><figcaption></figcaption></figure>

### Now lets look at one run in particular

Seed = 100

Initial Strategy Period = 1

rl sim portfolio value as of 2024-03-20 00:00:00: `12406271373.89722`

mvo sim portfolio value: 7624769419.518413 historical portfolio value as of 2024-03-20 00:00:00: $7,342,533,784.81

RL Cumualtive Return

22.57%

RL sortino ratio

0.16

MVO Cumulative Return

* 14.91%

MVO sortino ratio

* 0.04

Historical Cumulative Return

* \-34.42%

Historical sortino ratio

* \-0.55

<figure><img src=".gitbook/assets/rl_results.png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_eth_vault (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_steth_vault (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_btc_vault (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_alt_vault (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_psm_vault (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_lp_vault (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/rl_stable_vault (1).png" alt=""><figcaption></figcaption></figure>

### MVO Model Results

<figure><img src=".gitbook/assets/visualization (1) (1).png" alt=""><figcaption></figcaption></figure>

### Historical Comparison

<figure><img src=".gitbook/assets/visualization (2) (1).png" alt=""><figcaption></figcaption></figure>

### Comparisons

<figure><img src=".gitbook/assets/rl_vs_mvo_vs_historical_cum_return (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/robo_vs_historical_tvl (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/mvo_vs_rl_sortino (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/comp_vs_strat (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/newplot (22) (1).png" alt=""><figcaption></figcaption></figure>

### Robo Advisor Actions:

{% file src=".gitbook/assets/mvo_actions_final.csv" %}

{% file src=".gitbook/assets/rl_actions_final.csv" %}
