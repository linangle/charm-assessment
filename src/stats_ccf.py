"""
Cross-correlation, autocorrelation and stationarity (Anderson et al. 2016, Sec 2.5.1).

The paper's procedure, quoted:

  "Cross-correlation functions (CCF) were computed between time series of daily, modeled
   values and weekly in situ measurements at select shore stations to assess correlation
   and lead-lag relationships between discrete observations and predicted probability
   values. With CCF analysis, it is important not to violate assumptions of stationarity.
   In lieu of a formal test for stationarity (e.g. unit root test), stationarity in the
   time series was diagnosed by plotting the autocorrelation function (ACF) and assessing
   the steepness of the slope; a gradual decrease in the correlogram that does not taper
   to zero is a first order measure of a non-stationary process. If a linear trend exists
   in the time series, it is generally sufficient to detrend the data by simply removing
   the linear trend. In this case, the residuals of the linear regression of the data are
   used in the CCF analysis in place of the original time series. In cases of non-linear
   trends, stationarity can be achieved either by pre-whitening or by applying an
   Autoregressive Integrated Moving Average (ARIMA) model. An ARIMA model was applied to
   those data that fit this category by first differencing the time series and estimating
   the autoregressive, AR (p), integrated (d), and moving average, MA (q) parameters. The
   autocorrelation (ACF) and partial autocorrelation (PACF) plots of the differenced time
   series allow us to determine the type of ARIMA (either AR or MA) and number of terms
   that are needed for fitting the ARIMA (p,d,q) model (Box, Jenkins, and Reinsel, 1994).
   The residuals from the ARIMA fit are then cross-correlated with the second
   (observational) time series to attain the CCF."

Implementation notes
--------------------
TIME GRID. C-HARM v3.1 is missing ~20% of days (including a 53-day gap in Sep-Oct 2024),
and the observations are weekly. Both series are therefore placed on a common daily index
with NaN where absent, and every correlation is computed pairwise-complete. This mirrors
R's `ccf(..., na.action = na.pass)`, which is what the paper's R workflow would have
required given weekly observations against a daily model series.

LAG SIGN. Defined to match R's `ccf(x = model, y = obs)`, whose lag-k value estimates
cor(x[t+k], y[t]):
    CCF(k) = corr(model[t + k], obs[t])
      k < 0  ->  the model at an EARLIER time tracks the observation: model LEADS by |k| d
      k > 0  ->  the model at a LATER time tracks the observation:   model LAGS by k d
Every emitted row carries an explicit `interpretation` string so the sign cannot be
misread downstream.

STATIONARITY. The paper diagnosed stationarity by eye from the ACF. That diagnosis is
reproduced quantitatively (`acf_decay_diagnostic`) and, since it costs nothing and the
paper flags its own omission ("In lieu of a formal test ... e.g. unit root test"), it is
backed by both an ADF test (H0: unit root) and a KPSS test (H0: stationary). Reporting the
pair is standard practice because they fail in opposite directions.

PREWHITENING. The paper filters only the MODEL series through the ARIMA and cross-
correlates those residuals against the raw observations. Textbook prewhitening applies the
SAME filter to both series before correlating; filtering one side leaves the observation
side autocorrelated, which inflates the apparent significance of CCF spikes. The paper's
one-sided procedure is followed for comparability, and the two-sided (proper) variant is
computed alongside it so the difference is visible rather than assumed away.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from .config import ALPHA, MAX_LAG_DAYS


# Autocorrelation
def acf_pairwise(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(nlags + 1, np.nan)
    for k in range(nlags + 1):
        a = x[: len(x) - k] if k else x
        b = x[k:] if k else x
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3:
            va, vb = a[ok], b[ok]
            if va.std() > 0 and vb.std() > 0:
                out[k] = np.corrcoef(va, vb)[0, 1]
            else:
                out[k] = np.nan
    return out


def acf_decay_diagnostic(x: np.ndarray, nlags: int = MAX_LAG_DAYS) -> dict:
    x = np.asarray(x, dtype=float)
    n_eff = int(np.isfinite(x).sum())
    a = acf_pairwise(x, nlags)
    if n_eff < 10 or np.all(~np.isfinite(a[1:])):
        return dict(n_eff=n_eff, first_lag_below_1e=np.nan, first_lag_insignificant=np.nan,
                    frac_significant=np.nan, acf_slope=np.nan, nonstationary_by_acf=np.nan)

    bound = 1.96 / np.sqrt(n_eff)
    lags = np.arange(1, nlags + 1)
    vals = a[1:]

    below_1e = np.where(np.abs(vals) < 1 / np.e)[0]
    insig = np.where(np.abs(vals) < bound)[0]
    ok = np.isfinite(vals)
    slope = stats.linregress(lags[ok], vals[ok]).slope if ok.sum() >= 3 else np.nan

    return dict(
        n_eff=n_eff,
        first_lag_below_1e=int(lags[below_1e[0]]) if below_1e.size else np.nan,
        first_lag_insignificant=int(lags[insig[0]]) if insig.size else np.nan,
        frac_significant=float(np.nanmean(np.abs(vals) > bound)),
        acf_slope=float(slope) if np.isfinite(slope) else np.nan,
        nonstationary_by_acf=bool(insig.size == 0),
    )


def unit_root_tests(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 20:
        return dict(adf_stat=np.nan, adf_p=np.nan, kpss_stat=np.nan, kpss_p=np.nan,
                    verdict="insufficient data")

    # longest contiguous observed run
    runs, start = [], None
    for i, f in enumerate(np.append(finite, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return dict(adf_stat=np.nan, adf_p=np.nan, kpss_stat=np.nan, kpss_p=np.nan,
                    verdict="insufficient data")
    s, e = max(runs, key=lambda r: r[1] - r[0])
    seg = x[s:e]
    if seg.size < 20 or np.std(seg) == 0:
        return dict(adf_stat=np.nan, adf_p=np.nan, kpss_stat=np.nan, kpss_p=np.nan,
                    verdict="insufficient data")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_stat, adf_p = adfuller(seg, autolag="AIC")[:2]
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
        try:
            kpss_stat, kpss_p = kpss(seg, regression="c", nlags="auto")[:2]
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

    adf_stationary = np.isfinite(adf_p) and adf_p < ALPHA        # rejects unit root
    kpss_stationary = np.isfinite(kpss_p) and kpss_p >= ALPHA    # fails to reject stationarity
    if adf_stationary and kpss_stationary:
        verdict = "stationary (both agree)"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non-stationary (both agree)"
    elif adf_stationary and not kpss_stationary:
        verdict = "conflicting: ADF stationary, KPSS non-stationary (possible trend-stationarity)"
    else:
        verdict = "conflicting: ADF unit root, KPSS stationary (weak evidence)"

    return dict(
        adf_stat=float(adf_stat) if np.isfinite(adf_stat) else np.nan,
        adf_p=float(adf_p) if np.isfinite(adf_p) else np.nan,
        kpss_stat=float(kpss_stat) if np.isfinite(kpss_stat) else np.nan,
        kpss_p=float(kpss_p) if np.isfinite(kpss_p) else np.nan,
        n_segment=int(seg.size),
        verdict=verdict,
    )


def linear_trend(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return dict(r=np.nan, p=np.nan, slope=np.nan, significant=np.nan)
    res = stats.linregress(t[ok], x[ok])
    return dict(r=float(res.rvalue), p=float(res.pvalue), slope=float(res.slope),
                significant=bool(res.pvalue < ALPHA))


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """Residuals of the linear regression on time, NaNs preserved in place."""
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return x.copy()
    res = stats.linregress(t[ok], x[ok])
    return x - (res.intercept + res.slope * t)


# ARIMA
def select_arima(x: np.ndarray, max_p: int = 2, max_q: int = 2, d: int = 1) -> tuple:
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() < 30:
        return (0, d, 0), None

    best_aic, best_order, best_fit = np.inf, (0, d, 0), None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(x, order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit(method_kwargs={"warn_convergence": False})
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
            except Exception:
                continue
    return best_order, best_fit


def arima_residuals(x: np.ndarray, order: tuple | None = None) -> tuple[np.ndarray, tuple]:
    x = np.asarray(x, dtype=float)
    if order is None:
        order, fit = select_arima(x)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(x, order=order, enforce_stationarity=False,
                            enforce_invertibility=False).fit(method_kwargs={"warn_convergence": False})
        except Exception:
            fit = None
    if fit is None:
        return np.full_like(x, np.nan), order

    resid = np.asarray(fit.resid, dtype=float)
    resid[~np.isfinite(x)] = np.nan
    resid[: order[1]] = np.nan
    return resid, order


# cross-correlation
def ccf_pairwise(model: np.ndarray, obs: np.ndarray, max_lag: int = MAX_LAG_DAYS) -> pd.DataFrame:
    model = np.asarray(model, dtype=float)
    obs = np.asarray(obs, dtype=float)
    n = model.size
    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            m = model[k:]
            o = obs[: n - k]
        else:
            m = model[: n + k]
            o = obs[-k:]
        ok = np.isfinite(m) & np.isfinite(o)
        npair = int(ok.sum())
        if npair >= 5 and np.std(m[ok]) > 0 and np.std(o[ok]) > 0:
            r = float(np.corrcoef(m[ok], o[ok])[0, 1])
            bound = 1.96 / np.sqrt(npair)
        else:
            r, bound = np.nan, np.nan
        rows.append(dict(
            lag=k, r=r, n_pairs=npair, ci95=bound,
            significant=bool(np.isfinite(r) and abs(r) > bound),
            interpretation=("model leads obs by %d d" % -k) if k < 0
                           else ("model lags obs by %d d" % k) if k > 0
                           else "zero lag",
        ))
    return pd.DataFrame(rows)


def analyze_ccf(
    dates: pd.DatetimeIndex,
    model_daily: pd.Series,
    obs_series: pd.Series,
    max_lag: int = MAX_LAG_DAYS,
) -> dict:
    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    m = model_daily.reindex(idx)
    o = obs_series.reindex(idx)

    acf_diag = acf_decay_diagnostic(m.values, nlags=max_lag)
    roots = unit_root_tests(m.values)
    trend = linear_trend(m.values)

    nonstat_acf = acf_diag.get("nonstationary_by_acf")
    nonstat_root = "non-stationary" in str(roots.get("verdict", ""))
    order = None

    if trend["significant"] and not (nonstat_acf is True or nonstat_root):
        transform = "linear detrend"
        m_t = detrend_linear(m.values)
    elif (nonstat_acf is True) or nonstat_root or trend["significant"]:
        m_t, order = arima_residuals(m.values)
        transform = f"ARIMA{order} residuals"
    else:
        transform = "none (stationary)"
        m_t = m.values.copy()

    acf_after = acf_decay_diagnostic(m_t, nlags=max_lag)

    # CCF (following Anderson 2016)
    ccf = ccf_pairwise(m_t, o.values, max_lag=max_lag)

    # CCF with both series filtered
    if order is not None:
        o_t, _ = arima_residuals(o.values, order=order)
        ccf_two_sided = ccf_pairwise(m_t, o_t, max_lag=max_lag)
    else:
        ccf_two_sided = ccf

    sig = ccf[ccf["significant"]].copy().sort_values("r", key=np.abs, ascending=False)

    return dict(
        n_model_days=int(np.isfinite(m.values).sum()),
        n_obs=int(np.isfinite(o.values).sum()),
        span_days=len(idx),
        acf_before=acf_diag,
        acf_after=acf_after,
        unit_root=roots,
        trend=trend,
        transform=transform,
        arima_order=order,
        ccf=ccf,
        ccf_two_sided=ccf_two_sided,
        significant_peaks=sig,
        peak_lag=int(sig.iloc[0]["lag"]) if len(sig) else None,
        peak_r=float(sig.iloc[0]["r"]) if len(sig) else None,
        peak_interpretation=str(sig.iloc[0]["interpretation"]) if len(sig) else None,
        r_at_zero_lag=float(ccf.loc[ccf["lag"] == 0, "r"].iloc[0]),
    )
