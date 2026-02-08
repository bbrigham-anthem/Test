#!/usr/bin/env python3
"""PV10 (Present Value at 10% discount) calculator for decline curve reserves.

Uses WTI strip/forecast pricing, operating costs, and the DCA forecast
to compute a discounted cash flow valuation of remaining reserves.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

from dca.analysis import DeclineCurveAnalysis, DAYS_PER_MONTH


# ---------------------------------------------------------------------------
# WTI price deck — EIA Jan-2026 STEO quarterly + flat long-run tail
# Quarterly EIA forecasts ($/bbl):
#   Q1-2026: $54.93  Q2-2026: $52.67  Q3-2026: $52.03  Q4-2026: $49.34
#   Q1-2027: $49.00  Q2-2027: $50.66  Q3-2027: $50.68  Q4-2027: $51.00
# Beyond 2027: flat at $50/bbl (EIA long-run mid-case)
# ---------------------------------------------------------------------------

EIA_QUARTERLY = {
    (2026, 1): 54.93, (2026, 2): 52.67, (2026, 3): 52.03, (2026, 4): 49.34,
    (2027, 1): 49.00, (2027, 2): 50.66, (2027, 3): 50.68, (2027, 4): 51.00,
}
LONG_RUN_PRICE = 50.00  # $/bbl flat after 2027


def get_wti_price(date: pd.Timestamp) -> float:
    """Return WTI price ($/bbl) for a given month using EIA STEO + flat tail."""
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return EIA_QUARTERLY.get((year, quarter), LONG_RUN_PRICE)


def compute_pv10(
    forecast: pd.DataFrame,
    discount_rate: float = 0.10,
    opex_per_bbl: float = 15.00,
    severance_tax_rate: float = 0.046,
    price_deck: str = "eia_steo",
) -> pd.DataFrame:
    """Compute monthly net revenue and PV10 from a DCA forecast DataFrame.

    Parameters
    ----------
    forecast : DataFrame with columns [month, date, rate_per_day, cumulative]
    discount_rate : annual discount rate (default 10%)
    opex_per_bbl : operating expense per barrel (default $15)
    severance_tax_rate : production/severance tax as fraction of revenue (default 4.6%)
    price_deck : pricing source label

    Returns
    -------
    DataFrame with monthly cash flow detail and present values.
    """
    rows = []
    cumulative_pv = 0.0
    cumulative_ncf = 0.0

    for i, row in forecast.iterrows():
        month_num = row["month"]
        date = row["date"]
        rate = row["rate_per_day"]

        # Monthly production volume (bbl)
        monthly_vol = rate * DAYS_PER_MONTH

        # Revenue
        price = get_wti_price(date)
        gross_revenue = monthly_vol * price

        # Deductions
        severance_tax = gross_revenue * severance_tax_rate
        opex = monthly_vol * opex_per_bbl

        # Net cash flow
        ncf = gross_revenue - severance_tax - opex
        cumulative_ncf += ncf

        # Discount factor: 1 / (1 + r)^(t in years)
        years_from_now = month_num / 12.0
        discount_factor = 1.0 / (1 + discount_rate) ** years_from_now

        pv = ncf * discount_factor
        cumulative_pv += pv

        rows.append({
            "month": int(month_num),
            "date": date,
            "rate_bpd": round(rate, 1),
            "monthly_vol_bbl": round(monthly_vol, 0),
            "wti_price": round(price, 2),
            "gross_revenue": round(gross_revenue, 0),
            "severance_tax": round(severance_tax, 0),
            "opex": round(opex, 0),
            "net_cash_flow": round(ncf, 0),
            "discount_factor": round(discount_factor, 6),
            "pv_cash_flow": round(pv, 0),
            "cumulative_pv10": round(cumulative_pv, 0),
        })

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PV10 Reserve Valuation")
    parser.add_argument("csv", help="Production CSV file")
    parser.add_argument("--discount-rate", type=float, default=0.10, help="Annual discount rate (default: 0.10)")
    parser.add_argument("--opex", type=float, default=15.0, help="Operating cost $/bbl (default: 15)")
    parser.add_argument("--severance-tax", type=float, default=0.046, help="Severance tax rate (default: 0.046)")
    parser.add_argument("--economic-limit", type=float, default=5.0, help="Min economic rate bbl/day (default: 5)")
    parser.add_argument("--forecast-months", type=int, default=360, help="Forecast horizon months (default: 360)")
    parser.add_argument("--output", type=str, default=None, help="Save detailed cash flow CSV")
    args = parser.parse_args(argv)

    # Load data
    df = pd.read_csv(args.csv, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    rate_col = next(c for c in ("oil_rate", "gas_rate", "rate") if c in df.columns)
    df = df[df[rate_col] > 0].dropna(subset=[rate_col])

    # Run DCA
    dca = DeclineCurveAnalysis(df["date"], df[rate_col].values, economic_limit=args.economic_limit)
    best = dca.best_fit()
    reserves = dca.estimate_reserves()
    forecast = dca.forecast(months=args.forecast_months)

    # Compute PV10
    cf = compute_pv10(
        forecast,
        discount_rate=args.discount_rate,
        opex_per_bbl=args.opex,
        severance_tax_rate=args.severance_tax,
    )

    pv10 = cf["cumulative_pv10"].iloc[-1] if len(cf) > 0 else 0
    total_revenue = cf["gross_revenue"].sum()
    total_ncf = cf["net_cash_flow"].sum()
    total_vol = cf["monthly_vol_bbl"].sum()

    # Print summary
    print("=" * 65)
    print("PV10 RESERVE VALUATION")
    print("=" * 65)
    print(f"  Decline model:           {best.model.name} (R²={best.r_squared:.4f})")
    print(f"  qi = {best.model.qi:.1f} bbl/day, Di = {best.model.di:.6f}/mo, b = {best.model.b:.4f}")
    print()

    print("  PRICE DECK: EIA Jan-2026 STEO + $50/bbl flat tail")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  Q1-26: $54.93   Q2-26: $52.67   Q3-26: $52.03 │")
    print("  │  Q4-26: $49.34   Q1-27: $49.00   Q2-27: $50.66 │")
    print("  │  Q3-27: $50.68   Q4-27: $51.00   2028+: $50.00 │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    info = [
        ["Discount rate", f"{args.discount_rate:.0%}"],
        ["Operating cost", f"${args.opex:.2f}/bbl"],
        ["Severance tax rate", f"{args.severance_tax:.1%}"],
        ["Economic limit", f"{args.economic_limit} bbl/day"],
        ["", ""],
        ["Remaining reserves", f"{reserves.remaining_reserves:,.0f} bbl"],
        ["Forecast volume", f"{total_vol:,.0f} bbl"],
        ["Forecast life", f"{len(cf)} months"],
        ["", ""],
        ["Gross revenue (undiscounted)", f"${total_revenue:,.0f}"],
        ["Net cash flow (undiscounted)", f"${total_ncf:,.0f}"],
        ["", ""],
        [">>> PV10", f"${pv10:,.0f}"],
    ]
    print(tabulate(info, tablefmt="plain"))

    # Show annual summary
    if len(cf) > 0:
        cf["year"] = cf["date"].apply(lambda d: d.year)
        annual = cf.groupby("year").agg(
            avg_rate=("rate_bpd", "mean"),
            volume=("monthly_vol_bbl", "sum"),
            avg_price=("wti_price", "mean"),
            revenue=("gross_revenue", "sum"),
            ncf=("net_cash_flow", "sum"),
            pv=("pv_cash_flow", "sum"),
        ).reset_index()

        print("\n" + "=" * 65)
        print("ANNUAL SUMMARY")
        print("=" * 65)
        # Show first 10 years + last year
        if len(annual) > 12:
            show = pd.concat([annual.head(10), annual.tail(1)])
        else:
            show = annual

        rows = []
        for _, r in show.iterrows():
            rows.append([
                int(r["year"]),
                f"{r['avg_rate']:.0f}",
                f"{r['volume']:,.0f}",
                f"${r['avg_price']:.2f}",
                f"${r['revenue']:,.0f}",
                f"${r['ncf']:,.0f}",
                f"${r['pv']:,.0f}",
            ])
        print(tabulate(
            rows,
            headers=["Year", "Avg bpd", "Volume (bbl)", "Avg WTI", "Revenue", "Net CF", "PV @ 10%"],
            tablefmt="simple",
        ))

    if args.output:
        cf.to_csv(args.output, index=False)
        print(f"\nDetailed cash flow saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
