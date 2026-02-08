"""Plotting utilities for decline curve analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from dca.models import ExponentialDecline, HarmonicDecline, HyperbolicDecline
from dca.analysis import DeclineCurveAnalysis, DAYS_PER_MONTH


def _months_to_dates(origin: pd.Timestamp, t_months: np.ndarray) -> list:
    """Convert an array of months-from-origin to datetime objects."""
    return [origin + pd.Timedelta(days=float(m) * DAYS_PER_MONTH) for m in t_months]


def plot_decline_curves(
    dca: DeclineCurveAnalysis,
    forecast_months: int = 360,
    output_path: str | Path = "decline_curve.png",
) -> Path:
    """Plot historical data, all fitted models, and the forecast.

    Saves to *output_path* and returns the Path.
    """
    output_path = Path(output_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("Decline Curve Analysis", fontsize=14, fontweight="bold")

    # ----- Top: Rate vs Time -----
    ax1.scatter(
        dca.dates,
        dca.rates,
        s=20,
        color="black",
        zorder=5,
        label="Historical",
    )

    colors = {"Exponential": "#1f77b4", "Hyperbolic": "#ff7f0e", "Harmonic": "#2ca02c"}
    best = dca.best_fit()
    origin = dca.dates[0]

    for name, fit in dca._fits.items():
        model = fit.model
        t_end = dca.t_months[-1] + forecast_months
        t_fine = np.linspace(0, t_end, 500)
        rates_fine = model.rate(t_fine)
        dates_fine = _months_to_dates(origin, t_fine)
        style = "-" if model is best.model else "--"
        lw = 2.0 if model is best.model else 1.0
        label = f"{name} (R²={fit.r_squared:.4f})"
        if model is best.model:
            label += " *"
        ax1.plot(dates_fine, rates_fine, style, color=colors[name], lw=lw, label=label)

    # Economic limit line
    ax1.axhline(
        dca.economic_limit,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Economic limit ({dca.economic_limit:.0f})",
    )

    ax1.set_ylabel("Production Rate (bbl/day or mcf/day)")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    # ----- Bottom: Semi-log rate plot (common in DCA) -----
    ax2.scatter(dca.dates, dca.rates, s=20, color="black", zorder=5, label="Historical")
    for name, fit in dca._fits.items():
        model = fit.model
        t_end = dca.t_months[-1] + forecast_months
        t_fine = np.linspace(0, t_end, 500)
        rates_fine = model.rate(t_fine)
        dates_fine = _months_to_dates(origin, t_fine)
        style = "-" if model is best.model else "--"
        lw = 2.0 if model is best.model else 1.0
        ax2.plot(dates_fine, rates_fine, style, color=colors[name], lw=lw, label=name)

    ax2.axhline(dca.economic_limit, color="red", linestyle=":", alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel("Rate (log scale)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
