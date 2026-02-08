"""Arps decline curve models: Exponential, Hyperbolic, and Harmonic.

References:
    Arps, J.J. (1945). "Analysis of Decline Curves."
    Transactions of the AIME, 160(01), 228-247.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ExponentialDecline:
    """Exponential (constant-percentage) decline: b = 0.

    q(t) = qi * exp(-Di * t)
    Np(t) = (qi - q(t)) / Di          (cumulative production)
    """

    name = "Exponential"

    def __init__(self, qi: float, di: float) -> None:
        if qi <= 0:
            raise ValueError("qi must be positive")
        if di <= 0:
            raise ValueError("di must be positive")
        self.qi = qi
        self.di = di
        self.b = 0.0

    def rate(self, t: NDArray | float) -> NDArray:
        """Production rate at time *t* (same time units as Di)."""
        t = np.asarray(t, dtype=float)
        return self.qi * np.exp(-self.di * t)

    def cumulative(self, t: NDArray | float) -> NDArray:
        """Cumulative production from 0 to *t*."""
        q = self.rate(t)
        return (self.qi - q) / self.di

    def time_to_rate(self, q: float) -> float:
        """Time to reach rate *q*."""
        if q <= 0 or q > self.qi:
            raise ValueError("q must be between 0 and qi")
        return -np.log(q / self.qi) / self.di

    @staticmethod
    def curve_func(t: NDArray, qi: float, di: float) -> NDArray:
        """Static form for curve_fit: q(t) = qi * exp(-di * t)."""
        return qi * np.exp(-di * t)

    def __repr__(self) -> str:
        return f"ExponentialDecline(qi={self.qi:.2f}, Di={self.di:.6f})"


class HyperbolicDecline:
    """Hyperbolic decline: 0 < b < 1 (general Arps).

    q(t) = qi / (1 + b * Di * t)^(1/b)
    Np(t) = (qi^b / ((1-b)*Di)) * (qi^(1-b) - q(t)^(1-b))
    """

    name = "Hyperbolic"

    def __init__(self, qi: float, di: float, b: float) -> None:
        if qi <= 0:
            raise ValueError("qi must be positive")
        if di <= 0:
            raise ValueError("di must be positive")
        if not (0 < b < 1):
            raise ValueError("b must be between 0 and 1 (exclusive)")
        self.qi = qi
        self.di = di
        self.b = b

    def rate(self, t: NDArray | float) -> NDArray:
        t = np.asarray(t, dtype=float)
        return self.qi / np.power(1 + self.b * self.di * t, 1.0 / self.b)

    def cumulative(self, t: NDArray | float) -> NDArray:
        q = self.rate(t)
        return (
            np.power(self.qi, self.b)
            / ((1 - self.b) * self.di)
            * (np.power(self.qi, 1 - self.b) - np.power(q, 1 - self.b))
        )

    def time_to_rate(self, q: float) -> float:
        if q <= 0 or q > self.qi:
            raise ValueError("q must be between 0 and qi")
        return (np.power(self.qi / q, self.b) - 1) / (self.b * self.di)

    @staticmethod
    def curve_func(t: NDArray, qi: float, di: float, b: float) -> NDArray:
        return qi / np.power(1 + b * di * t, 1.0 / b)

    def __repr__(self) -> str:
        return (
            f"HyperbolicDecline(qi={self.qi:.2f}, Di={self.di:.6f}, b={self.b:.4f})"
        )


class HarmonicDecline:
    """Harmonic decline: b = 1 (special case of hyperbolic).

    q(t) = qi / (1 + Di * t)
    Np(t) = (qi / Di) * ln(qi / q(t))
    """

    name = "Harmonic"

    def __init__(self, qi: float, di: float) -> None:
        if qi <= 0:
            raise ValueError("qi must be positive")
        if di <= 0:
            raise ValueError("di must be positive")
        self.qi = qi
        self.di = di
        self.b = 1.0

    def rate(self, t: NDArray | float) -> NDArray:
        t = np.asarray(t, dtype=float)
        return self.qi / (1 + self.di * t)

    def cumulative(self, t: NDArray | float) -> NDArray:
        q = self.rate(t)
        return (self.qi / self.di) * np.log(self.qi / q)

    def time_to_rate(self, q: float) -> float:
        if q <= 0 or q > self.qi:
            raise ValueError("q must be between 0 and qi")
        return (self.qi / q - 1) / self.di

    @staticmethod
    def curve_func(t: NDArray, qi: float, di: float) -> NDArray:
        return qi / (1 + di * t)

    def __repr__(self) -> str:
        return f"HarmonicDecline(qi={self.qi:.2f}, Di={self.di:.6f})"
