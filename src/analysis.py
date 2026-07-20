from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import (
    ALPHA, BOTH_STATIONS, CA_WIDE_PDA, CA_WIDE_PN, DATA_INTERIM, LEADS, MATCHUP_MODES,
    MAX_LAG_DAYS, MODE_ADAPTIVE, MODE_STRICT, PDA_THRESHOLD_NG_ML, PN_THRESHOLD_CELLS_L,
    RESULTS, STATIONS,
)
from .prepare import build_matchups, build_observations, load_charm_series
from .skill import (
    auc_ci_delong, max_accuracy_point, maximize_f1_mcc_exact,
    optimize_prediction_point, pr_analysis, prediction_point_sweep, roc_analysis,
)
from .stats_ccf import analyze_ccf
from .ks_test import ks_test

QUANTITY_LABEL = {"pn": "Pseudo-nitzschia spp.", "pda": "Particulate domoic acid"}
QUANTITY_THRESHOLD = {
    "pn": f">= {PN_THRESHOLD_CELLS_L:.0f} cells/L",
    "pda": f">= {PDA_THRESHOLD_NG_ML} ng/mL (500 ng/L)",
}

MIN_MATCHUPS = 10
MIN_EVENTS = 1


def _units(mode_matchups: pd.DataFrame) -> list[dict]:
    units = []
    for q in ("pn", "pda"):
        avail = set(mode_matchups.loc[mode_matchups.quantity == q, "station"].unique())

        for code in (BOTH_STATIONS if q == "pn" else BOTH_STATIONS):
            if code in avail:
                units.append(dict(kind="station", quantity=q, key=code,
                                  label=STATIONS[code]["name"], stations=[code]))

        for code in sorted(avail - set(BOTH_STATIONS)):
            units.append(dict(kind="station", quantity=q, key=code,
                              label=STATIONS[code]["name"], stations=[code]))

        batch = [c for c in (CA_WIDE_PN if q == "pn" else CA_WIDE_PDA) if c in avail]
        if batch:
            units.append(dict(
                kind="california_wide", quantity=q, key=f"CA_WIDE_{q.upper()}",
                label=f"All California HAB stations ({QUANTITY_LABEL[q]})",
                stations=batch,
            ))
    return units


def _daily_series(charm: pd.DataFrame, obs: pd.DataFrame, unit: dict, lead: int, mode: str):
    q = unit["quantity"]
    codes = unit["stations"]

    c = charm[(charm.quantity == q) & (charm.lead == lead) & (charm.station.isin(codes))]
    o = obs[(obs.quantity == q) & (obs.station.isin(codes))]
    if c.empty or o.empty:
        return None, None

    model = c.groupby("date")["model_prob"].mean().sort_index()
    observed = o.groupby("date")["obs_value"].mean().sort_index()

    lo = min(model.index.min(), observed.index.min())
    hi = max(model.index.max(), observed.index.max())
    idx = pd.date_range(lo, hi, freq="D")
    return model.reindex(idx), observed.reindex(idx)


def run_unit(unit: dict, matchups: pd.DataFrame, charm: pd.DataFrame,
             obs: pd.DataFrame, lead: int, mode: str) -> dict | None:
    q = unit["quantity"]
    m = matchups[
        (matchups["mode"] == mode) & (matchups.quantity == q)
        & (matchups.lead == lead) & (matchups.station.isin(unit["stations"]))
    ]
    if len(m) < MIN_MATCHUPS:
        return None

    obs_val = m["obs_value"].to_numpy(float)
    obs_evt = m["obs_event"].to_numpy(int)
    prob = m["model_prob"].to_numpy(float)
    n_events = int(obs_evt.sum())

    row = dict(
        mode=mode, quantity=q, quantity_label=QUANTITY_LABEL[q],
        threshold=QUANTITY_THRESHOLD[q], unit_kind=unit["kind"], unit=unit["key"],
        unit_label=unit["label"], n_stations=len(unit["stations"]), lead_days=lead,
        n_matchups=len(m), n_events=n_events, n_non_events=len(m) - n_events,
        event_rate=round(float(obs_evt.mean()), 4),
        date_start=str(m["date"].min())[:10], date_end=str(m["date"].max())[:10],
        mean_model_prob=round(float(np.mean(prob)), 4),
        median_model_prob=round(float(np.median(prob)), 4),
        calibration_gap=round(float(np.mean(prob) - obs_evt.mean()), 4),
        prob_sep_event_minus_nonevent=(
            round(float(prob[obs_evt == 1].mean() - prob[obs_evt == 0].mean()), 4)
            if 0 < n_events < len(m) else np.nan),
    )

    # KS Test
    ks = ks_test(obs_val, prob, alpha=ALPHA)
    row.update(
        ks_D=ks["D"],
        ks_Dcrit=ks["Dcrit"],
        ks_p=ks["p"],
        ks_reject=ks["reject"])

    # Skill ROC PR
    if n_events >= MIN_EVENTS and n_events < len(m):
        sweep = prediction_point_sweep(obs_evt, prob)
        roc = roc_analysis(obs_evt, prob)
        pr = pr_analysis(obs_evt, prob)
        ci = auc_ci_delong(obs_evt, prob)

        opt_far = optimize_prediction_point(sweep, "FAR", "POD")
        opt_for = optimize_prediction_point(sweep, "FOR", "POD")
        opt_miss = optimize_prediction_point(sweep, "miss_rate", "POD")
        opt_acc = optimize_prediction_point(sweep, "accuracy", "POD")
        best_acc = max_accuracy_point(sweep)

        f1mcc = maximize_f1_mcc_exact(obs_evt, prob)

        row.update(
            auc=roc["auc_trapezoid"],
            auc_se=ci["se"], auc_ci_low=ci["ci_low"], auc_ci_high=ci["ci_high"],
            auc_beats_random=bool(np.isfinite(ci["ci_low"]) and ci["ci_low"] > 0.5),
            auc_pr=pr["auc_pr"], auc_pr_baseline=pr["auc_pr_baseline"],
            auc_pr_lift=pr["auc_pr_lift"],
            f1_max=f1mcc["f1_max"], f1_max_pp=f1mcc["f1_max_pp"],
            mcc_max=f1mcc["mcc_max"], mcc_max_pp=f1mcc["mcc_max_pp"],
            n_distinct_thresholds=f1mcc["n_thresholds"],
            opt_pp_far_pod=opt_far["prediction_point"], opt_val_far_pod=opt_far["value"],
            opt_pp_for_pod=opt_for["prediction_point"], opt_val_for_pod=opt_for["value"],
            opt_pp_miss_pod=opt_miss["prediction_point"], opt_val_miss_pod=opt_miss["value"],
            opt_pp_acc_pod=opt_acc["prediction_point"], opt_val_acc_pod=opt_acc["value"],
            max_accuracy=best_acc["accuracy"], max_accuracy_pp=best_acc["prediction_point"],
        )

        for tag, pp in (("faropt", opt_far["prediction_point"]),
                        ("foropt", opt_for["prediction_point"]),
                        ("missopt", opt_miss["prediction_point"]),
                        ("f1opt", f1mcc["f1_max_pp"]),
                        ("mccopt", f1mcc["mcc_max_pp"]),
                        ("at50", 0.5)):
            if np.isfinite(pp):
                near = sweep.iloc[(sweep["prediction_point"] - pp).abs().argmin()]
                for met in ("accuracy", "POD", "FAR", "POFD", "BS", "miss_rate", "FOR",
                            "precision", "F1", "MCC"):
                    row[f"{met}_{tag}"] = float(near[met]) if np.isfinite(near[met]) else np.nan
        row["_sweep"] = sweep
        row["_roc"] = roc
        row["_pr"] = pr
    else:
        note = ("no observed events" if n_events == 0 else "all match-ups are events")
        row.update(auc=np.nan, auc_pr=np.nan, skill_note=note)

    # CCF
    model_daily, obs_daily = _daily_series(charm, obs, unit, lead, mode)
    if model_daily is not None and np.isfinite(model_daily.values).sum() >= 30:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cc = analyze_ccf(model_daily.index, model_daily, obs_daily, max_lag=MAX_LAG_DAYS)
        row.update(
            ccf_transform=cc["transform"],
            arima_order=str(cc["arima_order"]) if cc["arima_order"] else "",
            model_trend_r=cc["trend"]["r"], model_trend_p=cc["trend"]["p"],
            model_trend_sig=cc["trend"]["significant"],
            acf_first_insig_lag=cc["acf_before"]["first_lag_insignificant"],
            acf_slope=cc["acf_before"]["acf_slope"],
            acf_frac_sig=cc["acf_before"]["frac_significant"],
            nonstationary_by_acf=cc["acf_before"]["nonstationary_by_acf"],
            adf_p=cc["unit_root"]["adf_p"], kpss_p=cc["unit_root"]["kpss_p"],
            stationarity_verdict=cc["unit_root"]["verdict"],
            acf_first_insig_lag_after=cc["acf_after"]["first_lag_insignificant"],
            ccf_r_at_zero=cc["r_at_zero_lag"],
            ccf_peak_lag=cc["peak_lag"], ccf_peak_r=cc["peak_r"],
            ccf_peak_meaning=cc["peak_interpretation"],
            ccf_n_sig_lags=int(len(cc["significant_peaks"])),
        )
        row["_ccf"] = cc["ccf"]
        row["_ccf_two_sided"] = cc["ccf_two_sided"]
    return row


def main() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    obs = build_observations()

    all_rows, sweeps, rocs, prs, ccfs = [], [], [], [], []
    for mode in MATCHUP_MODES:
        matchups = build_matchups(mode, obs)
        charm = load_charm_series(mode).dropna(subset=["model_prob"])
        units = _units(matchups)
        print(f"\n=== mode={mode}: {len(units)} analysis units x {len(LEADS)} leads ===")

        for unit in units:
            for lead in LEADS:
                r = run_unit(unit, matchups, charm, obs, lead, mode)
                if r is None:
                    continue
                for key, store in (("_sweep", sweeps), ("_roc", rocs), ("_pr", prs),
                                   ("_ccf", ccfs)):
                    obj = r.pop(key, None)
                    if obj is None:
                        continue
                    meta = dict(mode=mode, quantity=r["quantity"], unit=r["unit"], lead_days=lead)
                    if key in ("_sweep", "_ccf"):
                        store.append(obj.assign(**meta))
                    elif key == "_roc":
                        rc = obj["roc_swept"]
                        if len(rc):
                            store.append(rc.assign(**meta))
                    else:
                        pc = obj["pr_curve"]
                        if len(pc):
                            store.append(pc.assign(**meta))
                r.pop("_ccf_two_sided", None)
                all_rows.append(r)
            print(f"  {unit['quantity']:4s} {unit['label'][:46]:46s} done")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(RESULTS / "summary_stats.csv", index=False)
    if sweeps:
        pd.concat(sweeps, ignore_index=True).to_csv(RESULTS / "contingency_sweeps.csv", index=False)
    if rocs:
        pd.concat(rocs, ignore_index=True).to_csv(RESULTS / "roc_curves.csv", index=False)
    if prs:
        pd.concat(prs, ignore_index=True).to_csv(RESULTS / "pr_curves.csv", index=False)
    if ccfs:
        pd.concat(ccfs, ignore_index=True).to_csv(RESULTS / "ccf_tables.csv", index=False)

    print(f"\nWrote {RESULTS}/summary_stats.csv ({len(summary)} rows)")
    return summary


if __name__ == "__main__":
    main()
