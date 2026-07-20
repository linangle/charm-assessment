from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .config import (
    CA_WIDE_PDA, CA_WIDE_PN, DATA_INTERIM, FIGURES, RESULTS, STATIONS,
)
from .prepare import build_observations
from .stats_ccf import (
    acf_decay_diagnostic, acf_pairwise, linear_trend, unit_root_tests,
)

WINDOW_START = "2022-11-01"
MAX_LAG_WEEKS = 56
SEASON_WEEKS = 52


def _weekly(series_daily: pd.Series, period: str) -> np.ndarray:
    if period == "window":
        series_daily = series_daily[series_daily.index >= WINDOW_START]
    if series_daily.empty:
        return np.array([])
    wk = series_daily.resample("W").mean()
    wk = wk.reindex(pd.date_range(wk.index.min(), wk.index.max(), freq="W"))
    return np.log10(wk.values + 1.0)


def observed_weekly(obs: pd.DataFrame, codes: list[str], quantity: str, period: str) -> np.ndarray:
    o = obs[(obs.quantity == quantity) & (obs.station.isin(codes))]
    if o.empty:
        return np.array([])
    daily = o.groupby("date")["obs_value"].mean().sort_index()
    return _weekly(daily, period)


def model_weekly(charm: pd.DataFrame, codes: list[str], quantity: str, period: str) -> np.ndarray:
    c = charm[(charm.quantity == quantity) & (charm.lead == 0) & (charm.station.isin(codes))]
    if c.empty:
        return np.array([])
    daily = c.groupby("date")["model_prob"].mean().sort_index()
    if period == "window":
        daily = daily[daily.index >= WINDOW_START]
    wk = daily.resample("W").mean()
    wk = wk.reindex(pd.date_range(wk.index.min(), wk.index.max(), freq="W"))
    return wk.values


def battery(x: np.ndarray) -> dict:
    n = int(np.isfinite(x).sum())
    if n < 20:
        return dict(n_weeks=n, acf_w1=np.nan, acf_w2=np.nan, acf_w4=np.nan,
                    acf_w26=np.nan, acf_w52=np.nan, acf_slope=np.nan,
                    first_insig_lag=np.nan, nonstationary_by_acf=np.nan,
                    trend_r=np.nan, trend_p=np.nan, adf_p=np.nan, kpss_p=np.nan,
                    verdict="insufficient data")
    a = acf_pairwise(x, MAX_LAG_WEEKS)
    diag = acf_decay_diagnostic(x, nlags=min(MAX_LAG_WEEKS, 30))
    roots = unit_root_tests(x)
    tr = linear_trend(x)
    return dict(
        n_weeks=n,
        acf_w1=a[1], acf_w2=a[2], acf_w4=a[4],
        acf_w26=a[26] if a.size > 26 else np.nan,
        acf_w52=a[52] if a.size > 52 else np.nan,
        acf_slope=diag["acf_slope"],
        first_insig_lag=diag["first_lag_insignificant"],
        nonstationary_by_acf=diag["nonstationary_by_acf"],
        trend_r=tr["r"], trend_p=tr["p"],
        adf_p=roots["adf_p"], kpss_p=roots["kpss_p"],
        verdict=roots["verdict"],
    )


def _gapfill(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x).interpolate(method="linear", limit_area="inside")
    return s.dropna().values


def seasonal_difference(x: np.ndarray, period: int = SEASON_WEEKS) -> np.ndarray:
    s = pd.Series(x)
    return (s - s.shift(period)).values


def stl_decompose(x: np.ndarray, period: int = SEASON_WEEKS):
    xf = _gapfill(x)
    if xf.size < 2 * period + 1:
        return None
    res = STL(xf, period=period, robust=True).fit()
    return res.trend, res.seasonal, res.resid, xf.size


def _strength(component: np.ndarray, resid: np.ndarray) -> float:
    denom = np.nanvar(component + resid)
    if denom <= 0:
        return np.nan
    return float(max(0.0, 1.0 - np.nanvar(resid) / denom))


def stl_battery(x: np.ndarray, period: int = SEASON_WEEKS) -> dict:
    dec = stl_decompose(x, period)
    if dec is None:
        out = battery(np.array([]))
        out.update(stl_trend_r=np.nan, stl_trend_p=np.nan,
                   seasonal_strength=np.nan, trend_strength=np.nan)
        return out
    trend, seasonal, resid, n_used = dec
    out = battery(resid)
    tr = linear_trend(trend)
    out.update(
        stl_trend_r=tr["r"], stl_trend_p=tr["p"],
        seasonal_strength=_strength(seasonal, resid),
        trend_strength=_strength(trend, resid),
    )
    return out


def _units() -> list[tuple[str, str, list[str]]]:
    units = [("All California (PN)", "pn", CA_WIDE_PN),
             ("All California (pDA)", "pda", CA_WIDE_PDA)]
    for code, meta in STATIONS.items():
        if meta["has_pn"]:
            units.append((meta["name"], "pn", [code]))
        if meta["has_pda"]:
            units.append((meta["name"], "pda", [code]))
    return units


def main() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    obs = build_observations()
    charm = pd.read_csv(DATA_INTERIM / "charm_station_series.csv", parse_dates=["date"])
    m = charm[["station", "lead", "date", "pn_prob_r1", "pda_prob_r1"]].copy()
    mm = pd.concat([
        m.rename(columns={"pn_prob_r1": "model_prob"}).assign(quantity="pn")[["station", "lead", "date", "model_prob", "quantity"]],
        m.rename(columns={"pda_prob_r1": "model_prob"}).assign(quantity="pda")[["station", "lead", "date", "model_prob", "quantity"]],
    ], ignore_index=True).dropna(subset=["model_prob"])

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, q, codes in _units():
            for period in ("full", "window"):
                x = observed_weekly(obs, codes, q, period)
                methods = {
                    "raw": battery(x),
                    "seasonal_diff": battery(seasonal_difference(x)),
                    "stl": stl_battery(x),
                }
                for method, res in methods.items():
                    res = dict(res)
                    res.update(unit=label, quantity=q, period=period,
                               method=method, n_stations=len(codes))
                    rows.append(res)

    df = pd.DataFrame(rows)
    order = ["unit", "quantity", "period", "method", "n_stations", "n_weeks",
             "acf_w1", "acf_w2", "acf_w4", "acf_w26", "acf_w52", "acf_slope",
             "first_insig_lag", "nonstationary_by_acf",
             "trend_r", "trend_p", "adf_p", "kpss_p", "verdict",
             "stl_trend_r", "stl_trend_p", "seasonal_strength", "trend_strength"]
    df = df.reindex(columns=order)
    out = RESULTS / "obs_stationarity.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(df)} rows)\n")

    pd.set_option("display.width", 260)
    for period in ("full", "window"):
        ca = df[(df.period == period) & df.unit.str.startswith("All California")]
        print(f"=== California-wide observed series, {period} record (weekly, log10(x+1)) ===")
        raw = ca[ca.method == "raw"]
        print("  raw:            " + " | ".join(
            f"{r.quantity}: ACF w1={r.acf_w1:.2f} w52={r.acf_w52:.2f}  ADF p={r.adf_p:.3f} KPSS p={r.kpss_p:.3f}  [{r.verdict}]"
            for _, r in raw.iterrows()))
        sd = ca[ca.method == "seasonal_diff"]
        print("  seasonal_diff:  " + " | ".join(
            f"{r.quantity}: ADF p={r.adf_p:.3f} KPSS p={r.kpss_p:.3f}  [{r.verdict}]"
            for _, r in sd.iterrows()))
        st = ca[ca.method == "stl"]
        print("  STL:            " + " | ".join(
            f"{r.quantity}: F_seas={r.seasonal_strength:.2f} F_trend={r.trend_strength:.2f}  "
            f"trend r={r.stl_trend_r:+.2f} p={r.stl_trend_p:.3f}  remainder [{r.verdict}]"
            for _, r in st.iterrows()))
        print()

    _figure(obs, mm)
    _stl_figure(obs)
    return df


def _figure(obs: pd.DataFrame, charm: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, (label, q, codes) in zip(
            axes, [("All California (PN)", "pn", CA_WIDE_PN),
                   ("All California (pDA)", "pda", CA_WIDE_PDA)]):
            xo = observed_weekly(obs, codes, q, "full")
            xm = model_weekly(charm, codes, q, "full")
            ao = acf_pairwise(xo, MAX_LAG_WEEKS)
            am = acf_pairwise(xm, MAX_LAG_WEEKS)
            lags = np.arange(MAX_LAG_WEEKS + 1)
            no = int(np.isfinite(xo).sum())
            band = 1.96 / np.sqrt(no)
            ax.stem(lags, ao, basefmt=" ", markerfmt="o", linefmt="tab:blue",
                    label=f"observed (n={no} wk)")
            ax.plot(lags, am, color="tab:red", lw=1.3, marker=".", ms=4,
                    label="C-HARM model (weekly)")
            ax.axhline(band, color="grey", ls="--", lw=0.8)
            ax.axhline(-band, color="grey", ls="--", lw=0.8)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(52, color="green", ls=":", lw=1)
            ax.text(52, ax.get_ylim()[1] * 0.92, "annual", color="green", fontsize=8, ha="center")
            ax.set_title(f"{label}: weekly autocorrelation")
            ax.set_xlabel("lag (weeks)")
            ax.set_ylabel("ACF")
            ax.legend(fontsize=8)
        fig.suptitle("calHABMAP observed vs. C-HARM model autocorrelation (full record, weekly)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = FIGURES / "obs_acf_weekly.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\nWrote {out}")


def _stl_figure(obs: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex="col")
        for col, (label, q, codes) in enumerate(
            [("All California (PN)", "pn", CA_WIDE_PN),
             ("All California (pDA)", "pda", CA_WIDE_PDA)]):
            x = observed_weekly(obs, codes, q, "full")
            dec = stl_decompose(x)
            if dec is None:
                continue
            trend, seasonal, resid, n = dec
            xf = _gapfill(x)
            t = np.arange(n)
            Fs = _strength(seasonal, resid)
            Ft = _strength(trend, resid)
            for row, (series, name, color) in enumerate([
                    (xf, "observed", "tab:blue"), (trend, "trend", "tab:red"),
                    (seasonal, "seasonal (annual)", "tab:green"),
                    (resid, "remainder", "tab:grey")]):
                ax = axes[row, col]
                ax.plot(t, series, color=color, lw=0.9)
                ax.set_ylabel(name, fontsize=8)
                if row == 0:
                    ax.set_title(f"{label}\nseasonal strength={Fs:.2f}, trend strength={Ft:.2f}",
                                 fontsize=9)
                if row == 3:
                    ax.set_xlabel("week index (full record)")
                ax.tick_params(labelsize=7)
        fig.suptitle("STL decomposition of calHABMAP observed series (weekly, log10(x+1), period=52)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = FIGURES / "obs_stl_decomp.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
