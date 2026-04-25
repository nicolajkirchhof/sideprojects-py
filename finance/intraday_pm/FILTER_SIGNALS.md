# Filter Signals — Research Findings

Generated: 2026-04-20  
Data period: 2020-01-01 -> 2026-01-01  
Symbols: IBDE40, IBGB100, IBES35, IBJP225, IBUS30, IBUS500, IBUST100  

Three studies: (1) weekday directional bias, (2) post-extreme-day drift, (3) prior-day-close proximity windows.

---

## Study 1: Thursday/Friday/Monday — Weekday Directional Probability

**hc_pct**: % of sessions where high exceeded prior close.  
**lc_pct**: % of sessions where low went below prior close.  
**Structure**: prior-day bar structure relative to the day before it —
`hh_hl` (higher high + higher low), `lh_ll` (lower high + lower low),
`inside` (inside bar: both HH and LL), `outside` (outside bar: neither).  

Baseline for both hc and lc is ~70–80% (most days touch the prior close from both sides).
Look for weekday × structure combinations where hc_pct or lc_pct deviates >5pp from that baseline.

### IBDE40

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 136 | 87.5% | 91.9% |
| Monday | inside | 28 | 71.4% | 75.0% |
| Monday | lh_ll | 100 | 86.0% | 78.0% |
| Monday | outside | 33 | 90.9% | 75.8% |
| Tuesday | hh_hl | 150 | 84.0% | 88.7% |
| Tuesday | inside | 14 | 100.0% | 85.7% |
| Tuesday | lh_ll | 93 | 90.3% | 84.9% |
| Tuesday | outside | 48 | 93.8% | 83.3% |
| Wednesday | hh_hl | 140 | 90.7% | 85.0% |
| Wednesday | inside | 36 | 94.4% | 88.9% |
| Wednesday | lh_ll | 105 | 89.5% | 83.8% |
| Wednesday | outside | 22 | 86.4% | 95.5% |
| Thursday | hh_hl | 128 | 86.7% | 87.5% |
| Thursday | inside | 32 | 90.6% | 93.8% |
| Thursday | lh_ll | 100 | 88.0% | 88.0% |
| Thursday | outside | 42 | 92.9% | 83.3% |
| Friday | hh_hl | 114 | 89.5% | 90.4% |
| Friday | inside | 40 | 90.0% | 90.0% |
| Friday | lh_ll | 102 | 88.2% | 86.3% |
| Friday | outside | 40 | 82.5% | 92.5% |

### IBGB100

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 117 | 82.1% | 89.7% |
| Monday | inside | 30 | 86.7% | 70.0% |
| Monday | lh_ll | 105 | 87.6% | 78.1% |
| Monday | outside | 25 | 100.0% | 84.0% |
| Tuesday | hh_hl | 138 | 88.4% | 81.2% |
| Tuesday | inside | 23 | 78.3% | 82.6% |
| Tuesday | lh_ll | 96 | 89.6% | 84.4% |
| Tuesday | outside | 48 | 83.3% | 91.7% |
| Wednesday | hh_hl | 141 | 90.8% | 80.9% |
| Wednesday | inside | 41 | 92.7% | 75.6% |
| Wednesday | lh_ll | 91 | 90.1% | 82.4% |
| Wednesday | outside | 33 | 97.0% | 84.8% |
| Thursday | hh_hl | 136 | 83.8% | 90.4% |
| Thursday | inside | 33 | 90.9% | 90.9% |
| Thursday | lh_ll | 100 | 78.0% | 91.0% |
| Thursday | outside | 35 | 88.6% | 97.1% |
| Friday | hh_hl | 107 | 85.0% | 87.9% |
| Friday | inside | 37 | 83.8% | 78.4% |
| Friday | lh_ll | 116 | 87.9% | 89.7% |
| Friday | outside | 37 | 94.6% | 86.5% |

### IBES35

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 125 | 88.8% | 83.2% |
| Monday | inside | 35 | 94.3% | 68.6% |
| Monday | lh_ll | 101 | 80.2% | 80.2% |
| Monday | outside | 37 | 97.3% | 73.0% |
| Tuesday | hh_hl | 148 | 91.2% | 85.1% |
| Tuesday | inside | 24 | 100.0% | 95.8% |
| Tuesday | lh_ll | 96 | 86.5% | 82.3% |
| Tuesday | outside | 39 | 89.7% | 89.7% |
| Wednesday | hh_hl | 145 | 93.1% | 81.4% |
| Wednesday | inside | 42 | 95.2% | 78.6% |
| Wednesday | lh_ll | 93 | 87.1% | 89.2% |
| Wednesday | outside | 25 | 92.0% | 88.0% |
| Thursday | hh_hl | 139 | 91.4% | 84.9% |
| Thursday | inside | 23 | 91.3% | 82.6% |
| Thursday | lh_ll | 104 | 84.6% | 92.3% |
| Thursday | outside | 38 | 84.2% | 89.5% |
| Friday | hh_hl | 125 | 88.0% | 85.6% |
| Friday | inside | 29 | 86.2% | 86.2% |
| Friday | lh_ll | 109 | 84.4% | 85.3% |
| Friday | outside | 33 | 90.9% | 87.9% |

### IBJP225

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 110 | 84.5% | 70.0% |
| Monday | inside | 11 | 54.5% | 81.8% |
| Monday | lh_ll | 89 | 75.3% | 67.4% |
| Monday | outside | 69 | 91.3% | 88.4% |
| Tuesday | hh_hl | 117 | 84.6% | 76.1% |
| Tuesday | inside | 61 | 90.2% | 93.4% |
| Tuesday | lh_ll | 85 | 82.4% | 68.2% |
| Tuesday | outside | 32 | 81.2% | 68.8% |
| Wednesday | hh_hl | 143 | 78.3% | 77.6% |
| Wednesday | inside | 37 | 81.1% | 91.9% |
| Wednesday | lh_ll | 91 | 85.7% | 78.0% |
| Wednesday | outside | 26 | 80.8% | 100.0% |
| Thursday | hh_hl | 121 | 80.2% | 78.5% |
| Thursday | inside | 27 | 81.5% | 92.6% |
| Thursday | lh_ll | 115 | 81.7% | 74.8% |
| Thursday | outside | 32 | 71.9% | 81.2% |
| Friday | hh_hl | 115 | 73.9% | 85.2% |
| Friday | inside | 25 | 88.0% | 88.0% |
| Friday | lh_ll | 108 | 82.4% | 71.3% |
| Friday | outside | 44 | 75.0% | 88.6% |

### IBUS30

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 102 | 88.2% | 82.4% |
| Monday | inside | 22 | 81.8% | 77.3% |
| Monday | lh_ll | 78 | 85.9% | 80.8% |
| Monday | outside | 86 | 98.8% | 93.0% |
| Tuesday | hh_hl | 107 | 89.7% | 89.7% |
| Tuesday | inside | 97 | 96.9% | 97.9% |
| Tuesday | lh_ll | 74 | 94.6% | 79.7% |
| Tuesday | outside | 29 | 89.7% | 89.7% |
| Wednesday | hh_hl | 122 | 93.4% | 88.5% |
| Wednesday | inside | 50 | 90.0% | 94.0% |
| Wednesday | lh_ll | 102 | 94.1% | 93.1% |
| Wednesday | outside | 34 | 97.1% | 100.0% |
| Thursday | hh_hl | 114 | 91.2% | 84.2% |
| Thursday | inside | 47 | 97.9% | 93.6% |
| Thursday | lh_ll | 96 | 90.6% | 92.7% |
| Thursday | outside | 47 | 89.4% | 100.0% |
| Friday | hh_hl | 108 | 95.4% | 93.5% |
| Friday | inside | 43 | 88.4% | 93.0% |
| Friday | lh_ll | 113 | 92.0% | 89.4% |
| Friday | outside | 33 | 90.9% | 93.9% |

### IBUS500

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 114 | 87.7% | 81.6% |
| Monday | inside | 20 | 85.0% | 85.0% |
| Monday | lh_ll | 81 | 84.0% | 81.5% |
| Monday | outside | 73 | 100.0% | 95.9% |
| Tuesday | hh_hl | 121 | 91.7% | 89.3% |
| Tuesday | inside | 92 | 96.7% | 95.7% |
| Tuesday | lh_ll | 69 | 92.8% | 82.6% |
| Tuesday | outside | 25 | 88.0% | 88.0% |
| Wednesday | hh_hl | 142 | 93.0% | 88.7% |
| Wednesday | inside | 52 | 96.2% | 92.3% |
| Wednesday | lh_ll | 91 | 95.6% | 87.9% |
| Wednesday | outside | 23 | 95.7% | 100.0% |
| Thursday | hh_hl | 131 | 92.4% | 85.5% |
| Thursday | inside | 42 | 90.5% | 95.2% |
| Thursday | lh_ll | 94 | 87.2% | 88.3% |
| Thursday | outside | 37 | 94.6% | 78.4% |
| Friday | hh_hl | 116 | 91.4% | 88.8% |
| Friday | inside | 47 | 91.5% | 95.7% |
| Friday | lh_ll | 99 | 91.9% | 91.9% |
| Friday | outside | 35 | 94.3% | 91.4% |

### IBUST100

| Weekday | Structure | N | hc% | lc% |
|---------|-----------|---|-----|-----|
| Monday | hh_hl | 101 | 91.1% | 79.2% |
| Monday | inside | 23 | 73.9% | 87.0% |
| Monday | lh_ll | 78 | 85.9% | 84.6% |
| Monday | outside | 86 | 97.7% | 93.0% |
| Tuesday | hh_hl | 126 | 97.6% | 88.9% |
| Tuesday | inside | 76 | 96.1% | 100.0% |
| Tuesday | lh_ll | 73 | 89.0% | 86.3% |
| Tuesday | outside | 32 | 90.6% | 84.4% |
| Wednesday | hh_hl | 157 | 95.5% | 92.4% |
| Wednesday | inside | 39 | 94.9% | 84.6% |
| Wednesday | lh_ll | 92 | 94.6% | 89.1% |
| Wednesday | outside | 20 | 90.0% | 90.0% |
| Thursday | hh_hl | 132 | 93.2% | 83.3% |
| Thursday | inside | 37 | 89.2% | 83.8% |
| Thursday | lh_ll | 100 | 90.0% | 88.0% |
| Thursday | outside | 35 | 91.4% | 88.6% |
| Friday | hh_hl | 119 | 92.4% | 91.6% |
| Friday | inside | 48 | 87.5% | 97.9% |
| Friday | lh_ll | 101 | 94.1% | 92.1% |
| Friday | outside | 29 | 89.7% | 96.6% |

---

## Study 2: Post-Extreme-Day Forward Returns

Extreme day = |daily return| > 2%.  
Forward return measured from extreme day's close to first available close ≥ N weeks later.  
Positive mean/median on 'down' extreme days = mean-reversion tendency; negative = momentum.

### IBDE40

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 72 | -0.18% | 1.41% |
| down | 2 | 72 | 0.94% | 2.40% |
| down | 3 | 72 | 1.93% | 3.68% |
| down | 4 | 72 | 2.76% | 4.46% |
| up | 1 | 57 | 0.11% | 0.03% |
| up | 2 | 57 | 1.41% | 2.18% |
| up | 3 | 57 | 2.20% | 3.09% |
| up | 4 | 57 | 2.67% | 2.93% |

### IBGB100

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 46 | -0.07% | 1.33% |
| down | 2 | 46 | 0.70% | 2.57% |
| down | 3 | 46 | 1.31% | 3.22% |
| down | 4 | 46 | 1.76% | 2.47% |
| up | 1 | 33 | -1.11% | -0.08% |
| up | 2 | 33 | -0.40% | 0.67% |
| up | 3 | 33 | 0.53% | 0.79% |
| up | 4 | 33 | 0.99% | 1.20% |

### IBES35

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 67 | -0.01% | 1.23% |
| down | 2 | 67 | 0.61% | 2.20% |
| down | 3 | 67 | 1.25% | 2.75% |
| down | 4 | 67 | 2.10% | 3.10% |
| up | 1 | 52 | 0.18% | 0.56% |
| up | 2 | 52 | 0.89% | 0.92% |
| up | 3 | 52 | 1.45% | 1.59% |
| up | 4 | 52 | 1.41% | 1.08% |

### IBJP225

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 73 | 0.03% | 0.59% |
| down | 2 | 73 | 0.70% | 1.34% |
| down | 3 | 73 | 1.04% | 1.32% |
| down | 4 | 73 | 1.95% | 1.47% |
| up | 1 | 90 | 0.63% | 0.69% |
| up | 2 | 89 | 1.29% | 1.13% |
| up | 3 | 89 | 1.41% | 1.48% |
| up | 4 | 89 | 1.83% | 1.96% |

### IBUS30

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 52 | 0.38% | 0.75% |
| down | 2 | 52 | 0.72% | 2.44% |
| down | 3 | 52 | 1.56% | 2.55% |
| down | 4 | 52 | 2.42% | 3.07% |
| up | 1 | 49 | -0.56% | 0.41% |
| up | 2 | 49 | 0.30% | 1.56% |
| up | 3 | 49 | 1.53% | 2.64% |
| up | 4 | 49 | 2.77% | 3.61% |

### IBUS500

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 67 | 0.34% | 0.41% |
| down | 2 | 67 | 0.97% | 2.61% |
| down | 3 | 67 | 1.84% | 2.91% |
| down | 4 | 67 | 2.71% | 3.18% |
| up | 1 | 51 | -0.79% | -0.35% |
| up | 2 | 51 | 0.16% | 1.77% |
| up | 3 | 51 | 1.39% | 2.15% |
| up | 4 | 51 | 2.35% | 3.26% |

### IBUST100

| Direction | Weeks | N | Mean fwd ret% | Median fwd ret% |
|-----------|-------|---|---------------|-----------------|
| down | 1 | 118 | 0.63% | 0.58% |
| down | 2 | 118 | 0.86% | 1.65% |
| down | 3 | 118 | 1.61% | 2.43% |
| down | 4 | 118 | 2.16% | 2.58% |
| up | 1 | 115 | -0.51% | 0.24% |
| up | 2 | 115 | 0.29% | 1.67% |
| up | 3 | 115 | 0.82% | 1.76% |
| up | 4 | 115 | 1.11% | 2.44% |

---

## Study 3: PDC Proximity per 30-Min Session Window

**dist_pct**: mean/median distance of the nearest 5-min bar high or low to the
prior day close (PDC), expressed as % of PDC. Distance = 0 when PDC was crossed
within the window.  
Low dist_pct = price reliably tests PDC in that window — useful as a filter entry cue.

### IBDE40

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 07:00 | 1129 | 0.199% | 0.074% |
| 07:30 | 1129 | 0.288% | 0.151% |
| 08:00 | 1502 | 0.331% | 0.159% |
| 08:30 | 1502 | 0.397% | 0.217% |
| 09:00 | 1502 | 0.430% | 0.248% |
| 09:30 | 1502 | 0.460% | 0.277% |
| 10:00 | 1503 | 0.475% | 0.293% |
| 10:30 | 1503 | 0.501% | 0.307% |
| 11:00 | 1503 | 0.510% | 0.306% |
| 11:30 | 1503 | 0.528% | 0.318% |
| 12:00 | 1503 | 0.538% | 0.333% |
| 12:30 | 1503 | 0.541% | 0.325% |
| 13:00 | 1503 | 0.573% | 0.337% |
| 13:30 | 1503 | 0.534% | 0.305% |
| 14:00 | 1503 | 0.568% | 0.329% |
| 14:30 | 1503 | 0.585% | 0.343% |
| 15:00 | 1503 | 0.611% | 0.385% |
| 15:30 | 1503 | 0.650% | 0.423% |
| 16:00 | 1502 | 0.680% | 0.434% |
| 16:30 | 1502 | 0.694% | 0.443% |

### IBGB100

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 07:00 | 1116 | 0.169% | 0.061% |
| 07:30 | 1116 | 0.251% | 0.135% |
| 08:00 | 1489 | 0.281% | 0.119% |
| 08:30 | 1489 | 0.340% | 0.183% |
| 09:00 | 1489 | 0.362% | 0.191% |
| 09:30 | 1489 | 0.388% | 0.213% |
| 10:00 | 1489 | 0.403% | 0.230% |
| 10:30 | 1489 | 0.419% | 0.238% |
| 11:00 | 1489 | 0.425% | 0.242% |
| 11:30 | 1489 | 0.438% | 0.256% |
| 12:00 | 1489 | 0.445% | 0.269% |
| 12:30 | 1489 | 0.451% | 0.247% |
| 13:00 | 1477 | 0.473% | 0.263% |
| 13:30 | 1477 | 0.442% | 0.235% |
| 14:00 | 1477 | 0.466% | 0.260% |
| 14:30 | 1477 | 0.478% | 0.277% |
| 15:00 | 1477 | 0.502% | 0.303% |
| 15:30 | 1477 | 0.522% | 0.322% |

### IBES35

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 07:00 | 1134 | 0.188% | 0.046% |
| 07:30 | 1135 | 0.305% | 0.166% |
| 08:00 | 1509 | 0.341% | 0.168% |
| 08:30 | 1509 | 0.421% | 0.260% |
| 09:00 | 1510 | 0.455% | 0.281% |
| 09:30 | 1510 | 0.486% | 0.304% |
| 10:00 | 1510 | 0.512% | 0.328% |
| 10:30 | 1510 | 0.543% | 0.373% |
| 11:00 | 1510 | 0.560% | 0.378% |
| 11:30 | 1510 | 0.581% | 0.404% |
| 12:00 | 1510 | 0.592% | 0.400% |
| 12:30 | 1510 | 0.596% | 0.385% |
| 13:00 | 1504 | 0.634% | 0.430% |
| 13:30 | 1504 | 0.603% | 0.401% |
| 14:00 | 1504 | 0.628% | 0.438% |
| 14:30 | 1504 | 0.643% | 0.445% |
| 15:00 | 1504 | 0.667% | 0.470% |
| 15:30 | 1504 | 0.692% | 0.504% |
| 16:00 | 1504 | 0.733% | 0.537% |
| 16:30 | 1504 | 0.743% | 0.545% |

### IBJP225

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 00:00 | 1412 | 0.322% | 0.009% |
| 00:30 | 1420 | 0.451% | 0.264% |
| 01:00 | 1425 | 0.502% | 0.321% |
| 01:30 | 1428 | 0.540% | 0.339% |
| 02:00 | 1430 | 0.598% | 0.391% |
| 02:30 | 1430 | 0.640% | 0.445% |
| 03:00 | 1433 | 0.662% | 0.458% |
| 03:30 | 1433 | 0.633% | 0.437% |
| 04:00 | 1434 | 0.682% | 0.487% |
| 04:30 | 1434 | 0.697% | 0.506% |
| 05:00 | 1434 | 0.697% | 0.501% |
| 05:30 | 1434 | 0.701% | 0.511% |
| 06:00 | 1434 | 0.733% | 0.526% |
| 06:30 | 525 | 0.541% | 0.397% |
| 07:00 | 525 | 0.551% | 0.404% |
| 07:30 | 525 | 0.548% | 0.395% |
| 08:00 | 525 | 0.541% | 0.387% |
| 08:30 | 525 | 0.570% | 0.413% |

### IBUS30

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 13:00 | 1474 | 0.340% | 0.175% |
| 13:30 | 1485 | 0.298% | 0.113% |
| 14:00 | 1485 | 0.371% | 0.179% |
| 14:30 | 1485 | 0.388% | 0.197% |
| 15:00 | 1485 | 0.432% | 0.229% |
| 15:30 | 1485 | 0.478% | 0.275% |
| 16:00 | 1485 | 0.503% | 0.296% |
| 16:30 | 1485 | 0.526% | 0.316% |
| 17:00 | 1485 | 0.541% | 0.322% |
| 17:30 | 1482 | 0.568% | 0.338% |
| 18:00 | 1482 | 0.577% | 0.345% |
| 18:30 | 1473 | 0.597% | 0.368% |
| 19:00 | 1473 | 0.609% | 0.373% |
| 19:30 | 1473 | 0.606% | 0.392% |
| 20:00 | 821 | 0.555% | 0.378% |
| 20:30 | 821 | 0.562% | 0.378% |

### IBUS500

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 13:00 | 1473 | 0.374% | 0.212% |
| 13:30 | 1484 | 0.322% | 0.141% |
| 14:00 | 1484 | 0.386% | 0.182% |
| 14:30 | 1484 | 0.398% | 0.186% |
| 15:00 | 1484 | 0.440% | 0.220% |
| 15:30 | 1485 | 0.485% | 0.268% |
| 16:00 | 1485 | 0.516% | 0.282% |
| 16:30 | 1485 | 0.542% | 0.302% |
| 17:00 | 1485 | 0.561% | 0.322% |
| 17:30 | 1482 | 0.590% | 0.339% |
| 18:00 | 1482 | 0.598% | 0.351% |
| 18:30 | 1473 | 0.626% | 0.384% |
| 19:00 | 1473 | 0.643% | 0.393% |
| 19:30 | 1473 | 0.638% | 0.387% |
| 20:00 | 821 | 0.590% | 0.369% |
| 20:30 | 821 | 0.606% | 0.392% |

### IBUST100

| Window (UTC) | N | Mean dist% | Median dist% |
|-------------|---|------------|--------------|
| 13:00 | 1473 | 0.457% | 0.278% |
| 13:30 | 1485 | 0.370% | 0.163% |
| 14:00 | 1485 | 0.485% | 0.264% |
| 14:30 | 1485 | 0.492% | 0.250% |
| 15:00 | 1485 | 0.570% | 0.317% |
| 15:30 | 1485 | 0.633% | 0.374% |
| 16:00 | 1485 | 0.677% | 0.429% |
| 16:30 | 1485 | 0.717% | 0.466% |
| 17:00 | 1485 | 0.739% | 0.483% |
| 17:30 | 1482 | 0.773% | 0.499% |
| 18:00 | 1482 | 0.783% | 0.501% |
| 18:30 | 1473 | 0.820% | 0.527% |
| 19:00 | 1473 | 0.850% | 0.554% |
| 19:30 | 1473 | 0.856% | 0.561% |
| 20:00 | 821 | 0.811% | 0.558% |
| 20:30 | 821 | 0.853% | 0.602% |

---

## Application to BT-4/BT-5 Strategies

### Weekday bias

No actionable directional edge found. hc and lc probabilities are uniformly
high (80–97%) across all weekday × structure combinations for all instruments.
Both the daily high and low exceed prior close on the vast majority of sessions
regardless of day or prior structure — consistent with high-volatility index markets.
**Verdict: Do not apply a weekday directional filter to BT-4/BT-5 entries.**

### PDC proximity

Strong pattern: PDC is most likely to be tested in the **first 30-min window** of each
session. Median distance at open (07:00 UTC for EU, 13:30 UTC for US) is 0.06–0.18%
of price, vs 0.30–0.60% by mid-session. Distance grows monotonically through the day.

**Rule for BT-4/BT-5:** If the strategy entry signal fires within the first 30 min
of the session, treat it as higher-confidence (PDC test likely nearby). If entry
fires after the first 60 min, require PDC to have already been crossed to avoid
chasing a move that has extended too far from the reference level.

### Post-extreme-day drift

Consistent mean-reversion tendency 2–4 weeks after a down extreme day (|ret| > 2%
and negative): positive mean forward return across all 7 instruments. After up
extreme days, week-1 is slightly negative (momentum fade) before recovering.

**Rule for DRIFT:** After a down extreme day on an index underlying (IBUS500,
IBUST100, IBDE40, IBGB100), the 2-week forward drift is positive (mean +0.7–1.0%).
This supports entering DRIFT short-puts within 2 days of a down extreme day.
After an up extreme day, wait at least 1 week before adding new short-put positions.

| Filter | Rule | Applies to |
|--------|------|------------|
| Weekday bias | No actionable edge — do not filter | — |
| PDC proximity | First 30-min window = highest-confidence entry; after 60 min require PDC already crossed | BT-4, BT-5 |
| Extreme-day mean reversion | Enter DRIFT short-puts within 2 days of down extreme day | DRIFT timing |
| Extreme-day fade | Avoid adding DRIFT positions within 1 week of up extreme day | DRIFT timing |

