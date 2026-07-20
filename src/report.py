from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODE_ADAPTIVE, MODE_STRICT, RESULTS

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 80)


def _fmt(df, cols, nd=3):
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].astype(float).round(nd)
    return df


def table_skill(summ, mode=MODE_STRICT):
    """Skill table: ROC-AUC, PR-AUC, F1, MCC, and the operating points, per lead."""
    d = summ[summ["mode"] == mode].copy()
    cols = ["quantity", "unit_kind", "unit_label", "lead_days", "n_matchups", "n_events",
            "event_rate", "mean_model_prob", "calibration_gap",
            "prob_sep_event_minus_nonevent",
            "auc", "auc_se", "auc_ci_low", "auc_ci_high", "auc_beats_random",
            "auc_pr", "auc_pr_baseline", "auc_pr_lift",
            "f1_max", "f1_max_pp", "mcc_max", "mcc_max_pp",
            "opt_pp_far_pod", "POD_faropt", "FAR_faropt", "FOR_faropt", "POFD_faropt", "accuracy_faropt",
            "opt_pp_for_pod", "POD_foropt", "FAR_foropt", "FOR_foropt", "POFD_foropt", "accuracy_foropt",
            "max_accuracy"]
    cols = [c for c in cols if c in d]
    t = _fmt(d[cols], [c for c in cols if c not in
                       ("quantity", "unit_kind", "unit_label", "lead_days", "n_matchups",
                        "n_events", "auc_beats_random")])
    t = t.sort_values(["quantity", "unit_kind", "unit_label", "lead_days"])
    t.to_csv(RESULTS / f"table_skill_{mode}.csv", index=False)
    return t


def table_tests(summ, mode=MODE_STRICT):
    """Trend / stationarity / K-S / CCF table."""
    d = summ[summ["mode"] == mode].copy()
    cols = ["quantity", "unit_label", "lead_days", "n_matchups",
            "model_trend_r", "model_trend_p", "model_trend_sig",
            "nonstationary_by_acf", "adf_p", "kpss_p", "stationarity_verdict",
            "ccf_transform", "arima_order",
            "ks_D", "ks_Dcrit", "ks_reject", "ks_D_minmax",
            "ccf_r_at_zero", "ccf_peak_lag", "ccf_peak_r", "ccf_peak_meaning", "ccf_n_sig_lags"]
    cols = [c for c in cols if c in d]
    t = _fmt(d[cols], ["model_trend_r", "model_trend_p", "adf_p", "kpss_p", "ks_D",
                       "ks_Dcrit", "ks_D_minmax", "ccf_r_at_zero", "ccf_peak_r"])
    t = t.sort_values(["quantity", "unit_label", "lead_days"])
    t.to_csv(RESULTS / f"table_tests_{mode}.csv", index=False)
    return t


def main():
    summ = pd.read_csv(RESULTS / "summary_stats.csv")

    for mode in (MODE_STRICT, MODE_ADAPTIVE):
        if mode not in summ["mode"].unique():
            continue
        print("\n" + "=" * 120)
        print(f"SKILL TABLE - mode={mode}  (CA-wide rows are the batched assessments)")
        print("=" * 120)
        sk = table_skill(summ, mode)
        table_tests(summ, mode)
        ca = sk[sk.unit_kind == "california_wide"]
        show = ["quantity", "unit_label", "lead_days", "n_events", "event_rate",
                "auc", "auc_ci_low", "auc_ci_high", "auc_pr", "auc_pr_lift",
                "f1_max", "mcc_max", "opt_pp_far_pod", "opt_pp_for_pod"]
        print(ca[[c for c in show if c in ca]].to_string(index=False))

    print(f"\nWrote result tables to {RESULTS}/")


if __name__ == "__main__":
    main()
