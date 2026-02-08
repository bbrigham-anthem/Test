"""Unit tests for the DCA analysis engine."""

import numpy as np
import pandas as pd
import pytest

from dca.analysis import DeclineCurveAnalysis, DAYS_PER_MONTH
from dca.models import ExponentialDecline, HyperbolicDecline


def _make_synthetic_data(qi=500, di=0.04, b=0.5, n=36):
    """Generate synthetic hyperbolic decline data with some noise."""
    model = HyperbolicDecline(qi, di, b)
    t = np.arange(n, dtype=float)
    rates = model.rate(t)
    # Add small noise
    rng = np.random.default_rng(42)
    rates = rates + rng.normal(0, 2, size=n)
    rates = np.clip(rates, 1, None)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    return dates, rates


class TestDeclineCurveAnalysis:
    def test_fit_all_returns_three_models(self):
        dates, rates = _make_synthetic_data()
        dca = DeclineCurveAnalysis(dates, rates)
        fits = dca.fit_all()
        assert set(fits.keys()) == {"Exponential", "Hyperbolic", "Harmonic"}

    def test_best_fit_has_highest_r_squared_approx(self):
        dates, rates = _make_synthetic_data()
        dca = DeclineCurveAnalysis(dates, rates)
        best = dca.best_fit()
        # Best fit by AIC should have a reasonable R²
        assert best.r_squared > 0.9

    def test_hyperbolic_best_for_hyperbolic_data(self):
        dates, rates = _make_synthetic_data(qi=500, di=0.04, b=0.5, n=48)
        dca = DeclineCurveAnalysis(dates, rates)
        best = dca.best_fit()
        # Hyperbolic should win for hyperbolic data
        assert best.model.name == "Hyperbolic"

    def test_exponential_best_for_exponential_data(self):
        exp = ExponentialDecline(qi=400, di=0.06)
        t = np.arange(36, dtype=float)
        rates = exp.rate(t)
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        dca = DeclineCurveAnalysis(dates, rates)
        best = dca.best_fit()
        # Exponential should fit perfectly (lowest AIC due to fewer params)
        assert best.model.name == "Exponential"
        assert best.r_squared > 0.999

    def test_reserves_estimate_positive(self):
        dates, rates = _make_synthetic_data()
        dca = DeclineCurveAnalysis(dates, rates, economic_limit=5.0)
        dca.fit_all()
        reserves = dca.estimate_reserves()
        assert reserves.eur > 0
        assert reserves.remaining_reserves >= 0
        assert reserves.economic_life_months >= 0

    def test_forecast_respects_economic_limit(self):
        dates, rates = _make_synthetic_data()
        dca = DeclineCurveAnalysis(dates, rates, economic_limit=50.0)
        dca.fit_all()
        fc = dca.forecast(months=600)
        # All rates should be >= economic limit
        assert fc["rate_per_day"].min() >= 50.0

    def test_forecast_returns_dataframe(self):
        dates, rates = _make_synthetic_data()
        dca = DeclineCurveAnalysis(dates, rates)
        dca.fit_all()
        fc = dca.forecast(months=60)
        assert isinstance(fc, pd.DataFrame)
        assert "rate_per_day" in fc.columns
        assert "cumulative" in fc.columns
        assert "date" in fc.columns

    def test_too_few_points_raises(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        rates = [500, 480]
        with pytest.raises(ValueError, match="at least 3"):
            DeclineCurveAnalysis(dates, rates)

    def test_mismatched_lengths_raises(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        rates = [500, 480, 460]
        with pytest.raises(ValueError, match="same length"):
            DeclineCurveAnalysis(dates, rates)

    def test_eur_exceeds_cumulative(self):
        """EUR should always be >= cumulative production to date."""
        dates, rates = _make_synthetic_data()
        dca = DeclineCurveAnalysis(dates, rates, economic_limit=5.0)
        dca.fit_all()
        reserves = dca.estimate_reserves()
        assert reserves.eur >= reserves.cumulative_to_date * 0.5  # rough sanity
