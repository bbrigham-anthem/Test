#!/usr/bin/env python3
"""CLI entry point for Decline Curve Analysis.

Usage:
    python dca.py <csv_file> [options]

Examples:
    python dca.py data/sample_well.csv
    python dca.py data/sample_well.csv --model hyperbolic --economic-limit 10
    python dca.py data/sample_well.csv --forecast-months 240 --output forecast.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from dca.analysis import DeclineCurveAnalysis
from dca.plotting import plot_decline_curves


def load_production_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV with 'date' and a rate column (oil_rate or gas_rate)."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Determine rate column
    rate_col = None
    for candidate in ("oil_rate", "gas_rate", "rate"):
        if candidate in df.columns:
            rate_col = candidate
            break
    if rate_col is None:
        raise ValueError(
            "CSV must contain a rate column named 'oil_rate', 'gas_rate', or 'rate'"
        )

    # Drop zero/negative and NaN
    df = df[df[rate_col] > 0].dropna(subset=[rate_col])
    return df, rate_col


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Oil & Gas Decline Curve Analysis — Reserve Estimation"
    )
    parser.add_argument("csv", help="Path to production CSV file")
    parser.add_argument(
        "--model",
        choices=["exponential", "hyperbolic", "harmonic", "best"],
        default="best",
        help="Decline model to use (default: best fit by AIC)",
    )
    parser.add_argument(
        "--economic-limit",
        type=float,
        default=5.0,
        help="Minimum economic rate in bbl/day or mcf/day (default: 5)",
    )
    parser.add_argument(
        "--forecast-months",
        type=int,
        default=360,
        help="Number of months to forecast (default: 360)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save forecast CSV",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="decline_curve.png",
        help="Path to save decline curve plot (default: decline_curve.png)",
    )
    args = parser.parse_args(argv)

    # Load data
    print(f"Loading production data from {args.csv} ...")
    df, rate_col = load_production_data(args.csv)
    print(f"  {len(df)} data points loaded  ({df['date'].min().date()} to {df['date'].max().date()})")

    unit = "mcf/day" if rate_col == "gas_rate" else "bbl/day"

    # Run analysis
    dca = DeclineCurveAnalysis(
        dates=df["date"],
        rates=df[rate_col].values,
        economic_limit=args.economic_limit,
    )
    fits = dca.fit_all()
    best = dca.best_fit()

    # ---- Model comparison table ----
    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    rows = []
    for name, fit in fits.items():
        marker = " ★" if fit.model is best.model else ""
        rows.append(
            [
                name + marker,
                f"{fit.model.qi:.1f}",
                f"{fit.model.di:.6f}",
                f"{fit.model.b:.4f}",
                f"{fit.r_squared:.6f}",
                f"{fit.aic:.1f}",
            ]
        )
    print(
        tabulate(
            rows,
            headers=["Model", f"qi ({unit})", "Di (1/mo)", "b", "R²", "AIC"],
            tablefmt="simple",
        )
    )

    # ---- Select model ----
    model_map = {
        "exponential": fits.get("Exponential"),
        "hyperbolic": fits.get("Hyperbolic"),
        "harmonic": fits.get("Harmonic"),
        "best": best,
    }
    chosen_fit = model_map[args.model]
    chosen_model = chosen_fit.model

    # ---- Reserves ----
    reserves = dca.estimate_reserves(chosen_model)

    print("\n" + "=" * 65)
    print(f"RESERVES ESTIMATE  ({chosen_model.name} model)")
    print("=" * 65)
    vol_unit = "mcf" if rate_col == "gas_rate" else "bbl"
    info = [
        ["Initial rate (qi)", f"{reserves.qi:,.1f} {unit}"],
        ["Nominal decline (Di)", f"{reserves.di_nominal:.6f} /month"],
        ["Effective annual decline", f"{reserves.di_effective_annual:.2%}"],
        ["Arps b-factor", f"{reserves.b:.4f}"],
        ["R²", f"{reserves.r_squared:.6f}"],
        ["Cum. production to date", f"{reserves.cumulative_to_date:,.0f} {vol_unit}"],
        ["Est. Ultimate Recovery (EUR)", f"{reserves.eur:,.0f} {vol_unit}"],
        ["Remaining reserves", f"{reserves.remaining_reserves:,.0f} {vol_unit}"],
        ["Economic life remaining", f"{reserves.economic_life_months:,.0f} months"],
        ["Economic limit", f"{args.economic_limit} {unit}"],
    ]
    print(tabulate(info, tablefmt="plain"))

    # ---- Forecast ----
    fc = dca.forecast(chosen_model, months=args.forecast_months)
    print(f"\nForecast: {len(fc)} months until economic limit or horizon")

    # Show first/last few rows
    if len(fc) > 10:
        preview = pd.concat([fc.head(5), fc.tail(5)])
    else:
        preview = fc
    print(
        tabulate(
            preview.values.tolist(),
            headers=list(fc.columns),
            tablefmt="simple",
            floatfmt=".1f",
        )
    )

    # ---- Save outputs ----
    if args.output:
        fc.to_csv(args.output, index=False)
        print(f"\nForecast saved to {args.output}")

    plot_path = plot_decline_curves(dca, args.forecast_months, args.plot)
    print(f"Plot saved to {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
