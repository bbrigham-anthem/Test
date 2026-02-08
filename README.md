# Decline Curve Analysis (DCA) — Reserve Estimation Tool

A Python tool for oil and gas production decline curve analysis using Arps
empirical models. Fits historical production data to exponential, hyperbolic,
and harmonic decline curves to forecast future production and estimate
remaining recoverable reserves (EUR).

## Models

| Model | Equation | b-factor |
|---|---|---|
| Exponential | q(t) = qi · e^(−Di·t) | b = 0 |
| Hyperbolic | q(t) = qi / (1 + b·Di·t)^(1/b) | 0 < b < 1 |
| Harmonic | q(t) = qi / (1 + Di·t) | b = 1 |

Where:
- **qi** — initial production rate (at t=0 of the decline period)
- **Di** — initial nominal decline rate (1/time)
- **b** — Arps hyperbolic exponent

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis on a CSV file
python dca.py data/sample_well.csv

# Specify economic limit and forecast horizon
python dca.py data/sample_well.csv --economic-limit 5 --forecast-months 360

# Force a specific model
python dca.py data/sample_well.csv --model hyperbolic

# Output results to a file
python dca.py data/sample_well.csv --output results.csv
```

## Input Format

CSV with columns `date` (YYYY-MM-DD) and `oil_rate` (bbl/day) or
`gas_rate` (mcf/day):

```csv
date,oil_rate
2020-01-01,450
2020-02-01,430
2020-03-01,412
...
```

## Output

- Best-fit model selection with parameter estimates (qi, Di, b)
- Estimated Ultimate Recovery (EUR)
- Remaining reserves from the last production date
- Production forecast table
- Decline curve plot (saved as PNG)
