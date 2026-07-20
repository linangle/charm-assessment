from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, LEADS, MODE_STRICT, RESULTS

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
LEAD_COLORS = {0: "#1b7837", 1: "#7fbf7b", 2: "#f1a340", 3: "#d73027"}


def _read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load():
    return (_read("summary_stats.csv"), _read("roc_curves.csv"),
            _read("contingency_sweeps.csv"), _read("ccf_tables.csv"),
            _read("pr_curves.csv"))

def fig_roc_grid(summ, roc, mode=MODE_STRICT):
    for quantity in ("pn", "pda"):
        units = (summ[(summ["mode"] == mode) & (summ.quantity == quantity)
                      & summ.auc.notna()]["unit"].unique())
        units = [u for u in units if not roc[(roc.unit == u) & (roc.quantity == quantity)
                                             & (roc["mode"] == mode)].empty]
        if not units:
            continue
        ncol = 3
        nrow = int(np.ceil(len(units) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)

        for k, unit in enumerate(units):
            ax = axes.flat[k]
            ax.set_visible(True)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
            for lead in LEADS:
                rc = roc[(roc.unit == unit) & (roc.quantity == quantity)
                         & (roc["mode"] == mode) & (roc.lead_days == lead)]
                meta = summ[(summ.unit == unit) & (summ.quantity == quantity)
                            & (summ["mode"] == mode) & (summ.lead_days == lead)]
                if rc.empty or meta.empty:
                    continue
                auc = meta.iloc[0]["auc"]
                rc = rc.sort_values("POFD")
                ax.plot(rc["POFD"], rc["POD"], color=LEAD_COLORS[lead], lw=1.4,
                        label=f"{lead}d AUC={auc:.2f}")
            m0 = summ[(summ.unit == unit) & (summ.quantity == quantity)
                      & (summ["mode"] == mode) & (summ.lead_days == 0)].iloc[0]
            ax.set_title(f"{m0['unit_label'][:34]}\n(n={int(m0['n_matchups'])}, "
                         f"{int(m0['n_events'])} events)", fontsize=8)
            ax.set_xlabel("POFD (1 - specificity)")
            ax.set_ylabel("POD (sensitivity)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=6.5, loc="lower right")
        label = "Pseudo-nitzschia" if quantity == "pn" else "Particulate DA"
        fig.suptitle(f"C-HARM v3.1 ROC by forecast lead - {label} ({mode})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = FIGURES / f"roc_grid_{quantity}_{mode}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


def fig_contingency(summ, sweep, mode=MODE_STRICT):
    for quantity in ("pn", "pda"):
        units = [u for u in summ[(summ["mode"] == mode) & (summ.quantity == quantity)
                                 & summ.auc.notna()]["unit"].unique()
                 if not sweep[(sweep.unit == u) & (sweep.quantity == quantity)
                              & (sweep["mode"] == mode)].empty]
        if not units:
            continue
        ncol = 3
        nrow = int(np.ceil(len(units) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        metrics = [("accuracy", "#333333"), ("POD", "#1b7837"), ("FAR", "#d73027"),
                   ("POFD", "#f1a340"), ("BS", "#762a83")]
        for k, unit in enumerate(units):
            ax = axes.flat[k]
            ax.set_visible(True)
            sw = sweep[(sweep.unit == unit) & (sweep.quantity == quantity)
                       & (sweep["mode"] == mode) & (sweep.lead_days == 0)].sort_values("prediction_point")
            meta = summ[(summ.unit == unit) & (summ.quantity == quantity)
                        & (summ["mode"] == mode) & (summ.lead_days == 0)].iloc[0]
            if sw.empty:
                continue
            for met, col in metrics:
                if met in sw:
                    y = sw[met].clip(upper=4) if met == "BS" else sw[met]
                    ax.plot(sw["prediction_point"], y, color=col, lw=1.2, label=met)
            for key, col, style, lab in (
                    ("opt_pp_far_pod", "steelblue", "-", "min FAR|POD"),
                    ("opt_pp_for_pod", "purple", "--", "min FOR|POD")):
                pp = meta.get(key, np.nan)
                if np.isfinite(pp):
                    ax.axvline(pp, color=col, ls=style, lw=1.1, alpha=0.85,
                               label=f"{lab}={pp:.2f}")
            ax.set_title(f"{meta['unit_label'][:34]}", fontsize=8)
            ax.set_xlabel("prediction point (cutoff)")
            ax.set_ylabel("metric")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        label = "Pseudo-nitzschia" if quantity == "pn" else "Particulate DA"
        fig.suptitle(f"C-HARM v3.1 contingency metrics vs cutoff (nowcast) - {label} ({mode})",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = FIGURES / f"contingency_{quantity}_{mode}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


def fig_ccf(summ, ccf, mode=MODE_STRICT):
    for quantity in ("pn", "pda"):
        units = [u for u in summ[(summ["mode"] == mode) & (summ.quantity == quantity)]["unit"].unique()
                 if not ccf[(ccf.unit == u) & (ccf.quantity == quantity)
                            & (ccf["mode"] == mode) & (ccf.lead_days == 0)].empty]
        if not units:
            continue
        ncol = 3
        nrow = int(np.ceil(len(units) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, unit in enumerate(units):
            ax = axes.flat[k]
            ax.set_visible(True)
            c = ccf[(ccf.unit == unit) & (ccf.quantity == quantity)
                    & (ccf["mode"] == mode) & (ccf.lead_days == 0)].sort_values("lag")
            meta = summ[(summ.unit == unit) & (summ.quantity == quantity)
                        & (summ["mode"] == mode) & (summ.lead_days == 0)]
            if c.empty:
                continue
            ax.stem(c["lag"], c["r"], basefmt=" ", markerfmt=".", linefmt="steelblue")
            ax.plot(c["lag"], c["ci95"], color="k", ls="--", lw=0.7)
            ax.plot(c["lag"], -c["ci95"], color="k", ls="--", lw=0.7)
            ax.axvline(0, color="grey", lw=0.6)
            title = meta.iloc[0]["unit_label"][:32] if not meta.empty else unit
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("lag (days)   [<0: model leads]")
            ax.set_ylabel("cross-corr")
        label = "Pseudo-nitzschia" if quantity == "pn" else "Particulate DA"
        fig.suptitle(f"C-HARM v3.1 CCF, nowcast vs observations - {label} ({mode})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = FIGURES / f"ccf_{quantity}_{mode}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


def fig_auc_vs_lead(summ, mode=MODE_STRICT):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, quantity, label in ((axes[0], "pn", "Pseudo-nitzschia"),
                                (axes[1], "pda", "Particulate DA")):
        d = summ[(summ["mode"] == mode) & (summ.quantity == quantity) & summ.auc.notna()]
        for unit in sorted(d["unit"].unique()):
            u = d[d.unit == unit].sort_values("lead_days")
            is_ca = u.iloc[0]["unit_kind"] == "california_wide"
            ax.plot(u["lead_days"], u["auc"], marker="o", lw=2.4 if is_ca else 1.0,
                    color="black" if is_ca else None,
                    zorder=5 if is_ca else 2,
                    label=u.iloc[0]["unit_label"][:26] + (" *" if is_ca else ""))
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_title(f"{label}: AUC vs forecast lead ({mode})")
        ax.set_xlabel("forecast lead (days)")
        ax.set_ylabel("AUC")
        ax.set_xticks(LEADS)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    fig.tight_layout()
    out = FIGURES / f"auc_vs_lead_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_skill_vs_lead(summ, mode=MODE_STRICT):
    panels = [("mcc_max", "max MCC", 0.0, "no skill (MCC=0)", (-0.15, 0.7)),
              ("auc_pr_lift", "PR-AUC lift (AP / base rate)", 1.0, "no skill (lift=1)", (0, 3.0))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for col, (quantity, qlabel) in enumerate((("pn", "Pseudo-nitzschia"),
                                              ("pda", "Particulate DA"))):
        for rowi, (metric, mlabel, ref, reflabel, ylim) in enumerate(panels):
            ax = axes[rowi, col]
            d = summ[(summ["mode"] == mode) & (summ.quantity == quantity) & summ[metric].notna()]
            for unit in sorted(d["unit"].unique()):
                u = d[d.unit == unit].sort_values("lead_days")
                is_ca = u.iloc[0]["unit_kind"] == "california_wide"
                noisy = int(u.iloc[0]["n_events"]) < 10
                ax.plot(u["lead_days"], u[metric], marker="o",
                        lw=2.6 if is_ca else (0.8 if noisy else 1.4),
                        color="black" if is_ca else None,
                        alpha=1.0 if is_ca else (0.4 if noisy else 0.9),
                        zorder=5 if is_ca else 2,
                        label=u.iloc[0]["unit_label"][:24] + (" *" if is_ca else ""))
            ax.axhline(ref, color="grey", ls="--", lw=1)
            ax.set_title(f"{qlabel}: {mlabel} vs lead ({mode})", fontsize=10)
            ax.set_xlabel("forecast lead (days)")
            ax.set_ylabel(mlabel)
            ax.set_xticks(LEADS)
            ax.set_ylim(*ylim)
            if rowi == 0 and col == 1:
                ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle("C-HARM v3.1 skill vs forecast lead - imbalance-robust metrics\n"
                 "(faint lines = <10 events, unreliable; bold black = California-wide)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGURES / f"skill_vs_lead_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_calibration(summ, mode=MODE_STRICT):
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    d = summ[(summ["mode"] == mode) & (summ.lead_days == 0) & summ.event_rate.notna()
             & summ.mean_model_prob.notna()]
    for quantity, col, mk in (("pn", "#1b7837", "o"), ("pda", "#d73027", "s")):
        dd = d[d.quantity == quantity]
        ax.scatter(dd["event_rate"], dd["mean_model_prob"], c=col, marker=mk, s=40,
                   alpha=0.8, label=("Pseudo-nitzschia" if quantity == "pn" else "Particulate DA"))
        for _, r in dd.iterrows():
            if r["unit_kind"] == "california_wide":
                ax.annotate(r["unit_label"][:14], (r["event_rate"], r["mean_model_prob"]),
                            fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect reliability (1:1)")
    ax.set_xlabel("observed event rate (fraction of match-ups)")
    ax.set_ylabel("mean C-HARM probability")
    ax.set_title(f"C-HARM v3.1 reliability, nowcast ({mode})\npoints above 1:1 = over-forecasting")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / f"calibration_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")



def fig_pr_grid(summ, pr, mode=MODE_STRICT):
    for quantity in ("pn", "pda"):
        units = [u for u in summ[(summ["mode"] == mode) & (summ.quantity == quantity)
                                 & summ.auc_pr.notna()]["unit"].unique()
                 if not pr[(pr.unit == u) & (pr.quantity == quantity)
                           & (pr["mode"] == mode)].empty]
        if not units:
            continue
        ncol = 3
        nrow = int(np.ceil(len(units) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, unit in enumerate(units):
            ax = axes.flat[k]
            ax.set_visible(True)
            for lead in LEADS:
                pc = pr[(pr.unit == unit) & (pr.quantity == quantity)
                        & (pr["mode"] == mode) & (pr.lead_days == lead)]
                meta = summ[(summ.unit == unit) & (summ.quantity == quantity)
                            & (summ["mode"] == mode) & (summ.lead_days == lead)]
                if pc.empty or meta.empty:
                    continue
                ap = meta.iloc[0]["auc_pr"]
                pc = pc.sort_values("recall")
                ax.plot(pc["recall"], pc["precision"], color=LEAD_COLORS[lead], lw=1.4,
                        label=f"{lead}d AP={ap:.2f}")
            m0 = summ[(summ.unit == unit) & (summ.quantity == quantity)
                      & (summ["mode"] == mode) & (summ.lead_days == 0)].iloc[0]
            base = m0["event_rate"]
            ax.axhline(base, color="k", ls="--", lw=0.9, alpha=0.7,
                       label=f"no-skill={base:.2f}")
            ax.set_title(f"{m0['unit_label'][:34]}\n(n={int(m0['n_matchups'])}, "
                         f"{int(m0['n_events'])} events)", fontsize=8)
            ax.set_xlabel("recall (POD)")
            ax.set_ylabel("precision (1 - FAR)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=6.5, loc="upper right")
        label = "Pseudo-nitzschia" if quantity == "pn" else "Particulate DA"
        fig.suptitle(f"C-HARM v3.1 Precision-Recall by forecast lead - {label} ({mode})",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = FIGURES / f"pr_grid_{quantity}_{mode}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


def fig_f1_mcc(summ, sweep, mode=MODE_STRICT):
    for quantity in ("pn", "pda"):
        units = [u for u in summ[(summ["mode"] == mode) & (summ.quantity == quantity)
                                 & summ.mcc_max.notna()]["unit"].unique()
                 if not sweep[(sweep.unit == u) & (sweep.quantity == quantity)
                              & (sweep["mode"] == mode)].empty]
        if not units:
            continue
        ncol = 3
        nrow = int(np.ceil(len(units) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, unit in enumerate(units):
            ax = axes.flat[k]
            ax.set_visible(True)
            sw = sweep[(sweep.unit == unit) & (sweep.quantity == quantity)
                       & (sweep["mode"] == mode) & (sweep.lead_days == 0)].sort_values("prediction_point")
            meta = summ[(summ.unit == unit) & (summ.quantity == quantity)
                        & (summ["mode"] == mode) & (summ.lead_days == 0)]
            if sw.empty or meta.empty:
                continue
            m0 = meta.iloc[0]
            ax.plot(sw["prediction_point"], sw["F1"], color="#1b7837", lw=1.3, label="F1")
            ax.plot(sw["prediction_point"], sw["MCC"], color="#762a83", lw=1.3, label="MCC")
            ax.axhline(0, color="grey", lw=0.6)
            for pp, val, col in ((m0.get("f1_max_pp"), m0.get("f1_max"), "#1b7837"),
                                 (m0.get("mcc_max_pp"), m0.get("mcc_max"), "#762a83")):
                if np.isfinite(pp):
                    ax.plot([pp], [val], "o", color=col, ms=5)
                    ax.axvline(pp, color=col, ls=":", lw=0.8, alpha=0.6)
            ax.set_title(f"{m0['unit_label'][:32]}\nF1max={m0.get('f1_max', np.nan):.2f} "
                         f"MCCmax={m0.get('mcc_max', np.nan):.2f}", fontsize=8)
            ax.set_xlabel("prediction point (cutoff)")
            ax.set_ylabel("score")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.35, 1.0)
            ax.legend(fontsize=6.5, loc="upper right")
        label = "Pseudo-nitzschia" if quantity == "pn" else "Particulate DA"
        fig.suptitle(f"C-HARM v3.1 F1 & MCC vs cutoff (nowcast) - {label} ({mode})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = FIGURES / f"f1_mcc_{quantity}_{mode}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


def fig_metric_comparison(summ, mode=MODE_STRICT):
    d = summ[(summ["mode"] == mode) & (summ.lead_days == 0) & summ.auc.notna()].copy()
    d = d.sort_values(["quantity", "event_rate"])
    labels = [f"{r.unit_label[:20]} ({r.quantity},{int(r.n_events)}ev)" for _, r in d.iterrows()]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.34 * len(d))))
    ax.plot(d["auc"], y, "o", color="#2166ac", label="ROC-AUC")
    ax.plot(d["mcc_max"], y, "s", color="#762a83", label="max MCC")
    ax.plot(d["f1_max"], y, "^", color="#1b7837", label="max F1")
    ax.plot(d["auc_pr"], y, "D", color="#d6604d", label="PR-AUC (AP)")
    ax.axvline(0.5, color="#2166ac", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("score")
    ax.set_xlim(-0.15, 1.0)
    ax.set_title(f"C-HARM v3.1 nowcast: discrimination metrics side by side ({mode})\n"
                 "ROC-AUC dashed line = random (0.5); MCC/F1 solid line = 0")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = FIGURES / f"metric_comparison_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summ, roc, sweep, ccf, pr = _load()
    print("Rendering figures ...")
    for mode in summ["mode"].unique():
        fig_roc_grid(summ, roc, mode)
        fig_pr_grid(summ, pr, mode)
        fig_contingency(summ, sweep, mode)
        fig_f1_mcc(summ, sweep, mode)
        fig_ccf(summ, ccf, mode)
        fig_auc_vs_lead(summ, mode)
        fig_skill_vs_lead(summ, mode)
        fig_metric_comparison(summ, mode)
        fig_calibration(summ, mode)
    print(f"Figures in {FIGURES}/")


if __name__ == "__main__":
    main()
