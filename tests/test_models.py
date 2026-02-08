"""Unit tests for Arps decline curve models."""

import numpy as np
import pytest

from dca.models import ExponentialDecline, HarmonicDecline, HyperbolicDecline


class TestExponentialDecline:
    def test_rate_at_zero(self):
        m = ExponentialDecline(qi=500, di=0.05)
        assert m.rate(0.0) == pytest.approx(500.0)

    def test_rate_declines(self):
        m = ExponentialDecline(qi=500, di=0.05)
        r1 = m.rate(1.0)
        r10 = m.rate(10.0)
        assert r1 < 500
        assert r10 < r1

    def test_cumulative_increases(self):
        m = ExponentialDecline(qi=500, di=0.05)
        c1 = m.cumulative(1.0)
        c10 = m.cumulative(10.0)
        assert c1 > 0
        assert c10 > c1

    def test_cumulative_at_zero(self):
        m = ExponentialDecline(qi=500, di=0.05)
        assert m.cumulative(0.0) == pytest.approx(0.0)

    def test_time_to_rate(self):
        m = ExponentialDecline(qi=500, di=0.05)
        t = m.time_to_rate(250)
        assert m.rate(t) == pytest.approx(250, rel=1e-6)

    def test_invalid_qi_raises(self):
        with pytest.raises(ValueError):
            ExponentialDecline(qi=-1, di=0.05)

    def test_invalid_di_raises(self):
        with pytest.raises(ValueError):
            ExponentialDecline(qi=500, di=0)

    def test_curve_func_matches_rate(self):
        m = ExponentialDecline(qi=500, di=0.05)
        t = np.array([0, 5, 10, 20])
        np.testing.assert_allclose(
            ExponentialDecline.curve_func(t, 500, 0.05), m.rate(t)
        )


class TestHyperbolicDecline:
    def test_rate_at_zero(self):
        m = HyperbolicDecline(qi=500, di=0.05, b=0.5)
        assert m.rate(0.0) == pytest.approx(500.0)

    def test_rate_declines(self):
        m = HyperbolicDecline(qi=500, di=0.05, b=0.5)
        r1 = m.rate(1.0)
        r10 = m.rate(10.0)
        assert r1 < 500
        assert r10 < r1

    def test_cumulative_at_zero(self):
        m = HyperbolicDecline(qi=500, di=0.05, b=0.5)
        assert m.cumulative(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_time_to_rate(self):
        m = HyperbolicDecline(qi=500, di=0.05, b=0.5)
        t = m.time_to_rate(100)
        assert m.rate(t) == pytest.approx(100, rel=1e-6)

    def test_b_out_of_range_raises(self):
        with pytest.raises(ValueError):
            HyperbolicDecline(qi=500, di=0.05, b=0)
        with pytest.raises(ValueError):
            HyperbolicDecline(qi=500, di=0.05, b=1)

    def test_curve_func_matches_rate(self):
        m = HyperbolicDecline(qi=500, di=0.05, b=0.5)
        t = np.array([0, 5, 10, 20])
        np.testing.assert_allclose(
            HyperbolicDecline.curve_func(t, 500, 0.05, 0.5), m.rate(t)
        )

    def test_higher_b_slower_decline(self):
        """Higher b → slower decline (more reserves)."""
        m_low = HyperbolicDecline(qi=500, di=0.05, b=0.3)
        m_high = HyperbolicDecline(qi=500, di=0.05, b=0.8)
        assert m_high.rate(50) > m_low.rate(50)


class TestHarmonicDecline:
    def test_rate_at_zero(self):
        m = HarmonicDecline(qi=500, di=0.05)
        assert m.rate(0.0) == pytest.approx(500.0)

    def test_rate_declines(self):
        m = HarmonicDecline(qi=500, di=0.05)
        r1 = m.rate(1.0)
        r10 = m.rate(10.0)
        assert r1 < 500
        assert r10 < r1

    def test_cumulative_at_zero(self):
        m = HarmonicDecline(qi=500, di=0.05)
        assert m.cumulative(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_time_to_rate(self):
        m = HarmonicDecline(qi=500, di=0.05)
        t = m.time_to_rate(100)
        assert m.rate(t) == pytest.approx(100, rel=1e-6)

    def test_b_equals_one(self):
        m = HarmonicDecline(qi=500, di=0.05)
        assert m.b == 1.0

    def test_harmonic_declines_slower_than_exponential(self):
        """Harmonic should always be above exponential for the same qi/Di."""
        exp = ExponentialDecline(qi=500, di=0.05)
        har = HarmonicDecline(qi=500, di=0.05)
        t = np.linspace(1, 100, 50)
        assert np.all(har.rate(t) > exp.rate(t))
