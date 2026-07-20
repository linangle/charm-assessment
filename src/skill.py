from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

from .config import PREDICTION_POINTS_STEP


# Contingency
def contingency(obs_event: np.ndarray, prob: np.ndarray, prediction_point: float) -> dict:
    obs_event = np.asarray(obs_event).astype(bool)
    pred = np.asarray(prob, dtype=float) >= prediction_point

    hits = int(np.sum(pred & obs_event))            # TP
    false_alarms = int(np.sum(pred & ~obs_event))   # FP
    misses = int(np.sum(~pred & obs_event))         # FN
    correct_neg = int(np.sum(~pred & ~obs_event))   # TN
    total = hits + false_alarms + misses + correct_neg

    def _div(num, den):
        return float(num) / float(den) if den > 0 else np.nan

    # MCC
    mcc_den = np.sqrt(
        float(hits + false_alarms) * float(hits + misses)
        * float(correct_neg + false_alarms) * float(correct_neg + misses)
    )
    mcc = ((hits * correct_neg - false_alarms * misses) / mcc_den) if mcc_den > 0 else 0.0

    precision = _div(hits, hits + false_alarms)          # = 1 - FAR
    recall = _div(hits, hits + misses)                   # = POD
    f1 = (_div(2 * precision * recall, precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else np.nan)

    return dict(
        prediction_point=float(prediction_point),
        hits=hits, false_alarms=false_alarms, misses=misses, correct_negatives=correct_neg,
        total=total,
        accuracy=_div(hits + correct_neg, total),
        POD=recall,
        FAR=_div(false_alarms, hits + false_alarms),
        POFD=_div(false_alarms, correct_neg + false_alarms),
        BS=_div(hits + false_alarms, hits + misses),
        miss_rate=_div(misses, hits + misses),
        FOR=_div(misses, correct_neg + misses),
        precision=precision,
        recall=recall,
        F1=f1,
        MCC=float(mcc),
    )


def prediction_point_sweep(
    obs_event: np.ndarray,
    prob: np.ndarray,
    step: float = PREDICTION_POINTS_STEP,
) -> pd.DataFrame:
    points = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    return pd.DataFrame([contingency(obs_event, prob, p) for p in points])


def optimize_prediction_point(sweep: pd.DataFrame, metric_a: str, metric_b: str) -> dict:
    a = sweep[metric_a].to_numpy(dtype=float)
    b = sweep[metric_b].to_numpy(dtype=float)
    pts = sweep["prediction_point"].to_numpy(dtype=float)
    diff = a - b
    ok = np.isfinite(diff)
    if ok.sum() < 2:
        return dict(prediction_point=np.nan, value=np.nan, metric_a=metric_a,
                    metric_b=metric_b, exact_crossing=False)

    idx = np.where(ok)[0]
    sign = np.sign(diff[idx])
    cross = np.where(np.diff(sign) != 0)[0]
    if cross.size:
        i = idx[cross[0]]
        j = idx[cross[0] + 1]
        # linear interpolation of the crossing between the two bracketing points
        d0, d1 = diff[i], diff[j]
        w = 0.0 if d1 == d0 else d0 / (d0 - d1)
        pp = pts[i] + w * (pts[j] - pts[i])
        val = a[i] + w * (a[j] - a[i])
        return dict(prediction_point=float(pp), value=float(val), metric_a=metric_a,
                    metric_b=metric_b, exact_crossing=True)

    i = idx[np.nanargmin(np.abs(diff[idx]))]
    return dict(prediction_point=float(pts[i]), value=float((a[i] + b[i]) / 2),
                metric_a=metric_a, metric_b=metric_b, exact_crossing=False)


def max_accuracy_point(sweep: pd.DataFrame) -> dict:
    acc = sweep["accuracy"].to_numpy(dtype=float)
    if not np.isfinite(acc).any():
        return dict(prediction_point=np.nan, accuracy=np.nan)
    i = int(np.nanargmax(acc))
    return dict(prediction_point=float(sweep["prediction_point"].iloc[i]),
                accuracy=float(acc[i]))


def maximize_metric(sweep: pd.DataFrame, metric: str) -> dict:
    v = sweep[metric].to_numpy(dtype=float)
    pts = sweep["prediction_point"].to_numpy(dtype=float)
    if not np.isfinite(v).any():
        return dict(prediction_point=np.nan, value=np.nan, metric=metric)
    i = int(np.nanargmax(v))
    return dict(prediction_point=float(pts[i]), value=float(v[i]), metric=metric)


def maximize_f1_mcc_exact(obs_event: np.ndarray, prob: np.ndarray) -> dict:
    obs_event = np.asarray(obs_event).astype(int)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(prob)
    obs_event, prob = obs_event[ok], prob[ok]

    n_pos, n_neg = int(obs_event.sum()), int((1 - obs_event).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(f1_max=np.nan, f1_max_pp=np.nan, mcc_max=np.nan, mcc_max_pp=np.nan,
                    n_thresholds=0)

    fpr, tpr, thr = roc_curve(obs_event, prob, drop_intermediate=False)
    tp = tpr * n_pos
    fp = fpr * n_neg
    fn = n_pos - tp
    tn = n_neg - fp

    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)

    def _argmax(vals):
        return int(np.nanargmax(vals)) if np.isfinite(vals).any() else 0

    i_f1 = _argmax(f1)
    i_mcc = int(np.nanargmax(mcc))
    f1_thr = float(min(thr[i_f1], 1.0)) if np.isfinite(f1[i_f1]) else np.nan
    mcc_thr = float(min(thr[i_mcc], 1.0))
    return dict(
        f1_max=float(f1[i_f1]) if np.isfinite(f1[i_f1]) else np.nan,
        f1_max_pp=f1_thr,
        mcc_max=float(mcc[i_mcc]),
        mcc_max_pp=mcc_thr,
        n_thresholds=int(thr.size),
    )


# ROC AUC
def roc_analysis(obs_event: np.ndarray, prob: np.ndarray, step: float = PREDICTION_POINTS_STEP) -> dict:
    obs_event = np.asarray(obs_event).astype(int)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(prob)
    obs_event, prob = obs_event[ok], prob[ok]

    n_pos, n_neg = int(obs_event.sum()), int((1 - obs_event).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(n=int(obs_event.size), n_events=n_pos, n_non_events=n_neg,
                    auc_trapezoid=np.nan,
                    roc_empirical=pd.DataFrame(), roc_swept=pd.DataFrame(),
                    note="AUC undefined: only one class present")

    fpr, tpr, thr = roc_curve(obs_event, prob, drop_intermediate=False)
    auc_trap = float(roc_auc_score(obs_event, prob))

    sweep = prediction_point_sweep(obs_event, prob, step=step)
    swept = sweep[["prediction_point", "POFD", "POD"]].dropna()

    return dict(
        n=int(obs_event.size),
        n_events=n_pos,
        n_non_events=n_neg,
        auc_trapezoid=auc_trap,
        roc_empirical=pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thr)),
        roc_swept=swept.reset_index(drop=True),
        note="",
    )

# PR AUC
def pr_analysis(obs_event: np.ndarray, prob: np.ndarray) -> dict:
    obs_event = np.asarray(obs_event).astype(int)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(prob)
    obs_event, prob = obs_event[ok], prob[ok]

    n_pos, n_neg = int(obs_event.sum()), int((1 - obs_event).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(auc_pr=np.nan, auc_pr_baseline=np.nan, auc_pr_lift=np.nan,
                    pr_curve=pd.DataFrame(), note="PR undefined: only one class present")

    precision, recall, thr = precision_recall_curve(obs_event, prob)
    ap = float(average_precision_score(obs_event, prob))
    base = float(obs_event.mean())
    curve = pd.DataFrame(dict(recall=recall, precision=precision))
    return dict(
        auc_pr=ap,
        auc_pr_baseline=base,
        auc_pr_lift=float(ap / base) if base > 0 else np.nan,
        pr_curve=curve,
        note="",
    )


def auc_ci_delong(obs_event: np.ndarray, prob: np.ndarray, alpha: float = 0.05) -> dict:
    obs_event = np.asarray(obs_event).astype(int)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(prob)
    obs_event, prob = obs_event[ok], prob[ok]
    n_pos, n_neg = int(obs_event.sum()), int((1 - obs_event).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(auc=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan)

    a = float(roc_auc_score(obs_event, prob))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    from scipy import stats as _st
    z = _st.norm.ppf(1 - alpha / 2)
    return dict(auc=a, se=se,
                ci_low=float(max(0.0, a - z * se)),
                ci_high=float(min(1.0, a + z * se)))
