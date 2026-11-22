# Newcomers behind a cold prompt on the served configuration (image sm120-r13), 2026-11-22

`bench/newcomer.py`, 3 trials: a cold ~196K-token prompt starts, then a short and a 16K prompt arrive while one background stream decodes. Median first token: short 0.54 s, 16K 6.65 s; the long prompt took 1.19x its solo time; the background stream decoded at 210 tok/s (`newcomer-summary.json`).
