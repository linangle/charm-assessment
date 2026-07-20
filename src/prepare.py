"""
Build observation / model match-ups.

Observations come from calHABMAP; predictions come from the extracted C-HARM station series

Output: data_interim/matchups.csv

Run:  python3 -m src.prepare
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BOX_RADIUS, BOX_RADIUS_WIDE, CALHABMAP_CSV, COL_PDA, COL_PN_DELICATISSIMA,
    COL_PN_SERIATA, DATA_INTERIM, MATCHUP_MODES, MODE_ADAPTIVE, MODE_STRICT,
    PDA_THRESHOLD_NG_ML, PN_THRESHOLD_CELLS_L, STATIONS,
)


def load_calhabmap() -> pd.DataFrame:
    df = pd.read_csv(CALHABMAP_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["time_utc"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df = df[df["location_code"].isin(STATIONS)].copy()

    # batch delicatissima and seriata into Pseudo nitzschia spp. -- note that seriata is generally considered the more toxic one
    d = df[COL_PN_DELICATISSIMA]
    s = df[COL_PN_SERIATA]
    both = d.notna() & s.notna()
    df["pn_cells_l"] = np.where(both, d.fillna(0) + s.fillna(0), np.nan)

    df["pda_ng_ml"] = df[COL_PDA]
    return df


def build_observations() -> pd.DataFrame:
    """Long-format observations: one row per (station, quantity, date)."""
    df = load_calhabmap()
    frames = []

    pn = df.dropna(subset=["pn_cells_l"])[["location_code", "date", "pn_cells_l"]].copy()
    pn = pn.rename(columns={"location_code": "station", "pn_cells_l": "obs_value"})
    pn["quantity"] = "pn"
    pn["obs_event"] = (pn["obs_value"] >= PN_THRESHOLD_CELLS_L).astype(int)
    frames.append(pn)

    pda = df.dropna(subset=["pda_ng_ml"])[["location_code", "date", "pda_ng_ml"]].copy()
    pda = pda.rename(columns={"location_code": "station", "pda_ng_ml": "obs_value"})
    pda["quantity"] = "pda"
    pda["obs_event"] = (pda["obs_value"] >= PDA_THRESHOLD_NG_ML).astype(int)
    frames.append(pda)

    obs = pd.concat(frames, ignore_index=True)

    # calHABMAP can have more than one record per station/day --> collapse to one observation per day
    # by the mean (single day cannot be double weighted)
    obs = (
        obs.groupby(["station", "quantity", "date"], as_index=False)
        .agg(obs_value=("obs_value", "mean"), n_samples=("obs_value", "size"))
    )
    obs["obs_event"] = np.where(
        obs["quantity"] == "pn",
        (obs["obs_value"] >= PN_THRESHOLD_CELLS_L).astype(int),
        (obs["obs_value"] >= PDA_THRESHOLD_NG_ML).astype(int),
    )
    return obs


def load_charm_series(mode: str) -> pd.DataFrame:
    if mode not in MATCHUP_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MATCHUP_MODES}")

    df = pd.read_csv(DATA_INTERIM / "charm_station_series.csv", parse_dates=["date"])
    frames = []
    for tag in ("pn", "pda"):
        narrow = df[f"{tag}_prob_r{BOX_RADIUS}"]
        if mode == MODE_STRICT:
            prob = narrow
            npix = df[f"{tag}_npix_r{BOX_RADIUS}"]
            radius = pd.Series(BOX_RADIUS, index=df.index).where(narrow.notna())
        else:
            wide = df[f"{tag}_prob_r{BOX_RADIUS_WIDE}"]
            prob = narrow.where(narrow.notna(), wide)
            npix = df[f"{tag}_npix_r{BOX_RADIUS}"].where(
                narrow.notna(), df[f"{tag}_npix_r{BOX_RADIUS_WIDE}"]
            )
            radius = pd.Series(BOX_RADIUS, index=df.index).where(
                narrow.notna(), BOX_RADIUS_WIDE
            ).where(prob.notna())

        sub = df[["station", "lead", "date"]].copy()
        sub["model_prob"] = prob
        sub["n_pixels"] = npix
        sub["box_radius"] = radius
        sub["quantity"] = tag
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def build_matchups(mode: str, obs: pd.DataFrame | None = None) -> pd.DataFrame:
    obs = build_observations() if obs is None else obs
    charm = load_charm_series(mode)

    m = obs.merge(charm, on=["station", "quantity", "date"], how="inner")
    m = m.dropna(subset=["model_prob"])
    m["station_name"] = m["station"].map(lambda c: STATIONS[c]["name"])
    m["mode"] = mode
    return m.sort_values(["quantity", "station", "lead", "date"]).reset_index(drop=True)


def main() -> pd.DataFrame:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    obs = build_observations()

    frames = [build_matchups(mode, obs) for mode in MATCHUP_MODES]
    m = pd.concat(frames, ignore_index=True)

    out = DATA_INTERIM / "matchups.csv"
    m.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(m):,} rows across modes {MATCHUP_MODES})\n")

    pd.set_option("display.width", 240)
    for mode in MATCHUP_MODES:
        summary = (
            m[(m["mode"] == mode) & (m.lead == 0)]
            .groupby(["quantity", "station", "station_name"])
            .agg(n=("obs_event", "size"), events=("obs_event", "sum"),
                 event_rate=("obs_event", "mean"), box=("box_radius", "max"),
                 start=("date", "min"), end=("date", "max"))
            .reset_index()
        )
        summary["event_rate"] = summary["event_rate"].round(3)
        summary["box"] = summary["box"].map({1: "3x3", 2: "5x5"})
        print(f"=== mode={mode} :: match-ups at lead 0 ===")
        print(summary.to_string(index=False))
        print()
    return m


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
