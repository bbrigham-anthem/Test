"""Decline Curve Analysis engine — fitting, model selection, and reserve estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from dca.models import ExponentialDecline, HarmonicDecline, HyperbolicDecline

# Days-per-month constant used to convert between daily rates and monthly time
DAYS_PER_MONTH = 30.4375


@dataclass
class FitResult:
    """Container for a single model-fit result."""

    model: ExponentialDecline | HyperbolicDecline | HarmonicDecline
    r_squared: float
    sse: float  # sum of squared errors
    aic: float  # Akaike information criterion


@dataclass
class ReservesEstimate:
    """Container for reserves / EUR output."""

    model_name: str
    qi: float  # bbl/day or mcf/day
    di_nominal: float  # 1/month
    di_effective_annual: float  # fraction/year
    b: float
    eur: float  # total estimated ultimate recovery (bbl or mcf)
    cumulative_to_date: float
    remaining_reserves: float
    economic_life_months: float
    r_squared: float


class DeclineCurveAnalysis:
    """Fit Arps decline models to production history and estimate reserves.

    Parameters
    ----------
    dates : array-like of datetime
        Production dates (one per period).
    rates : array-like of float
        Production rates (bbl/day or mcf/day).
    economic_limit : float
        Minimum economic rate — forecast stops here (default 5 bbl/day).
    """

    def __init__(
        self,
        dates: pd.Series | list,
        rates: pd.Series | list,
        economic_limit: float = 5.0,
    ) -> None:
        self.dates = pd.DatetimeIndex(pd.to_datetime(dates))
        self.rates = np.asarray(rates, dtype=float)

        if len(self.dates) != len(self.rates):
            raise ValueError("dates and rates must have the same length")
        if len(self.dates) < 3:
            raise ValueError("need at least 3 data points for curve fitting")

        self.economic_limit = economic_limit

        # Time axis in months from first date
        td = self.dates - self.dates[0]
        delta_days = np.array([x.total_seconds() / 86400 for x in td])
        self.t_months: NDArray = delta_days / DAYS_PER_MONTH

        # Cumulative production to date (trapezoidal, rate*days -> volume)
        delta_days_arr = np.diff(delta_days)
        avg_rates = (self.rates[:-1] + self.rates[1:]) / 2.0
        self.cum_production = np.sum(avg_rates * delta_days_arr)

        self._fits: dict[str, FitResult] = {}
        self._best: Optional[FitResult] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, observed: NDArray, predicted: NDArray, n_params: int
    ) -> tuple[float, float, float]:
        """Return (R², SSE, AIC)."""
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n = len(observed)
        aic = n * np.log(ss_res / n + 1e-30) + 2 * n_params
        return r2, ss_res, aic

    def fit_exponential(self) -> FitResult:
        try:
            popt, _ = curve_fit(
                ExponentialDecline.curve_func,
                self.t_months,
                self.rates,
                p0=[self.rates[0], 0.05],
                bounds=([0, 1e-8], [np.inf, 10.0]),
                maxfev=10000,
            )
            qi, di = popt
            model = ExponentialDecline(qi, di)
            predicted = model.rate(self.t_months)
            r2, sse, aic = self._compute_metrics(self.rates, predicted, 2)
            result = FitResult(model=model, r_squared=r2, sse=sse, aic=aic)
        except RuntimeError:
            # Fit failed — return a poor-quality placeholder
            model = ExponentialDecline(self.rates[0], 0.01)
            result = FitResult(model=model, r_squared=-1, sse=np.inf, aic=np.inf)
        self._fits["Exponential"] = result
        return result

    def fit_hyperbolic(self) -> FitResult:
        try:
            popt, _ = curve_fit(
                HyperbolicDecline.curve_func,
                self.t_months,
                self.rates,
                p0=[self.rates[0], 0.05, 0.5],
                bounds=([0, 1e-8, 0.01], [np.inf, 10.0, 0.99]),
                maxfev=10000,
            )
            qi, di, b = popt
            model = HyperbolicDecline(qi, di, b)
            predicted = model.rate(self.t_months)
            r2, sse, aic = self._compute_metrics(self.rates, predicted, 3)
            result = FitResult(model=model, r_squared=r2, sse=sse, aic=aic)
        except RuntimeError:
            model = HyperbolicDecline(self.rates[0], 0.01, 0.5)
            result = FitResult(model=model, r_squared=-1, sse=np.inf, aic=np.inf)
        self._fits["Hyperbolic"] = result
        return result

    def fit_harmonic(self) -> FitResult:
        try:
            popt, _ = curve_fit(
                HarmonicDecline.curve_func,
                self.t_months,
                self.rates,
                p0=[self.rates[0], 0.05],
                bounds=([0, 1e-8], [np.inf, 10.0]),
                maxfev=10000,
            )
            qi, di = popt
            model = HarmonicDecline(qi, di)
            predicted = model.rate(self.t_months)
            r2, sse, aic = self._compute_metrics(self.rates, predicted, 2)
            result = FitResult(model=model, r_squared=r2, sse=sse, aic=aic)
        except RuntimeError:
            model = HarmonicDecline(self.rates[0], 0.01)
            result = FitResult(model=model, r_squared=-1, sse=np.inf, aic=np.inf)
        self._fits["Harmonic"] = result
        return result

    def fit_all(self) -> dict[str, FitResult]:
        """Fit all three Arps models and return results dict."""
        self.fit_exponential()
        self.fit_hyperbolic()
        self.fit_harmonic()
        return self._fits

    def best_fit(self) -> FitResult:
        """Select the best model by lowest AIC (penalises extra parameters)."""
        if not self._fits:
            self.fit_all()
        self._best = min(self._fits.values(), key=lambda f: f.aic)
        return self._best

    # ------------------------------------------------------------------
    # Reserves / EUR
    # ------------------------------------------------------------------

    def estimate_reserves(
        self, model: Optional[ExponentialDecline | HyperbolicDecline | HarmonicDecline] = None
    ) -> ReservesEstimate:
        """Estimate EUR and remaining reserves for a given (or best-fit) model.

        Time axis is in *months*; rates are daily.
        EUR = cumulative_to_date + remaining_forecast (rate × days).
        """
        if model is None:
            if self._best is None:
                self.best_fit()
            fit = self._best
            model = fit.model
            r2 = fit.r_squared
        else:
            # Find matching fit result
            r2 = next(
                (f.r_squared for f in self._fits.values() if f.model is model), 0.0
            )

        # Time from t=0 to economic limit
        try:
            t_econ = model.time_to_rate(self.economic_limit)
        except ValueError:
            # economic limit is above qi — no remaining life
            t_econ = 0.0

        # EUR in "rate·months" then convert to barrels (rate is daily)
        eur_rate_months = model.cumulative(t_econ)  # integral in rate·months
        eur = eur_rate_months * DAYS_PER_MONTH  # convert to bbl (or mcf)

        # What's already been produced (in same units)
        t_last = self.t_months[-1]
        cum_model_to_date = model.cumulative(t_last) * DAYS_PER_MONTH

        remaining = max(eur - cum_model_to_date, 0.0)

        # Effective annual decline from nominal monthly
        di_eff_annual = 1 - (1 - model.di) ** 12 if model.di < 1 else 1.0

        return ReservesEstimate(
            model_name=model.name,
            qi=model.qi,
            di_nominal=model.di,
            di_effective_annual=di_eff_annual,
            b=model.b,
            eur=eur,
            cumulative_to_date=self.cum_production,
            remaining_reserves=remaining,
            economic_life_months=max(t_econ - t_last, 0.0),
            r_squared=r2,
        )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(
        self,
        model: Optional[ExponentialDecline | HyperbolicDecline | HarmonicDecline] = None,
        months: int = 360,
    ) -> pd.DataFrame:
        """Generate a monthly production forecast from the last historical date.

        Returns a DataFrame with columns: month, date, rate, cumulative.
        """
        if model is None:
            if self._best is None:
                self.best_fit()
            model = self._best.model

        t_last = self.t_months[-1]
        t_forecast = np.arange(0, months + 1, dtype=float)
        t_abs = t_last + t_forecast  # absolute time from t=0

        rates = model.rate(t_abs)
        cum = model.cumulative(t_abs) * DAYS_PER_MONTH

        # Truncate at economic limit
        mask = rates >= self.economic_limit
        rates = rates[mask]
        cum = cum[mask]
        t_forecast = t_forecast[: len(rates)]

        last_date = self.dates[-1]
        forecast_dates = [
            last_date + pd.DateOffset(months=int(m)) for m in t_forecast
        ]

        return pd.DataFrame(
            {
                "month": t_forecast.astype(int),
                "date": forecast_dates,
                "rate_per_day": np.round(rates, 2),
                "cumulative": np.round(cum, 0),
            }
        )
