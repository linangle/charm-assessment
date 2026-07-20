"""
Extracting C-HARM v3.1 station time series from the NetCDF files.

For every station and lead time (0 (nowcast) / 1 / 2 / 3 days), pull the box of grid cells centered on the cell
nearest the calHABMAP station's fixed coordinates and average the valid pixels, following Anderson et al. (2016).

Output: data_interim/charm_station_series.csv, tidy, one row per (station, lead, date):
    station, lead, date, {pn,pda}_prob_r{1,2}, {pn,pda}_npix_r{1,2}

Run:  python3 -m src.extract
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    BOX_RADII, CHARM_FILES, DATA_INTERIM, LEADS, MIN_VALID_PIXELS,
    STATIONS, VAR_PDA, VAR_PN,
)

def nearest_index(coord_values: np.ndarray, target: float) -> int:
    """Index of the grid coordinate closest to `target`"""
    return int(np.abs(coord_values - target).argmin())


def box_slice(idx: int, n: int, radius: int) -> slice:
    """A slice of +/- `radius` cells around `idx`, clipped to the array bounds"""
    return slice(max(idx - radius, 0), min(idx + radius + 1, n))


def extract_station_box(
    da: xr.DataArray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    station_lat: float,
    station_lon: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    i = nearest_index(lat_vals, station_lat)
    j = nearest_index(lon_vals, station_lon)

    box = da.isel(
        latitude=box_slice(i, lat_vals.size, radius),
        longitude=box_slice(j, lon_vals.size, radius),
    ).values

    box = box.reshape(box.shape[0], -1)
    n_valid = np.isfinite(box).sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.where(
            n_valid >= MIN_VALID_PIXELS,
            np.nanmean(np.where(np.isfinite(box), box, np.nan), axis=1),
            np.nan,
        )
    return means, n_valid


def extract_lead(lead: int) -> pd.DataFrame:
    """Extract all stations for one lead time."""
    path = CHARM_FILES[lead]
    print(f"  [lead {lead}] opening {path.name} ...", flush=True)

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lat_vals = ds.latitude.values
        lon_vals = ds.longitude.values
        dates = pd.DatetimeIndex(ds.time.values).normalize()

        # Read each variable's station boxes once at each radius
        rows = []
        for code, meta in STATIONS.items():
            out = {"station": code, "lead": lead, "date": dates}
            for var, tag in ((VAR_PN, "pn"), (VAR_PDA, "pda")):
                for radius in BOX_RADII:
                    means, npix = extract_station_box(
                        ds[var], lat_vals, lon_vals, meta["lat"], meta["lon"], radius
                    )
                    out[f"{tag}_prob_r{radius}"] = means
                    out[f"{tag}_npix_r{radius}"] = npix
            frame = pd.DataFrame(out)
            rows.append(frame)
            pn1 = np.isfinite(frame["pn_prob_r1"]).mean() * 100
            pda1 = np.isfinite(frame["pda_prob_r1"]).mean() * 100
            print(
                f"    {code:4s} {meta['name']:30s} 3x3 valid days: "
                f"PN {pn1:5.1f}%  pDA {pda1:5.1f}%",
                flush=True,
            )

    return pd.concat(rows, ignore_index=True)


def main() -> pd.DataFrame:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    print("Extracting C-HARM v3.1 3x3 box means at calHABMAP stations")
    frames = [extract_lead(lead) for lead in LEADS]
    df = pd.concat(frames, ignore_index=True)

    out = DATA_INTERIM / "charm_station_series.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df):,} rows)")

    # Report where the paper's 3x3 protocol returns ~nothing -- station/quantity is considered unusable at <5% valid days
    # check whether 5x5 box would be usable in such case
    print("\nCoverage audit (lead 0, % of C-HARM days with a value):")
    lead0 = df[df.lead == 0]
    print(f"  {'stn':4s} {'name':30s} {'PN 3x3':>7s} {'PN 5x5':>7s} {'pDA 3x3':>8s} {'pDA 5x5':>8s}")
    for code, meta in STATIONS.items():
        s = lead0[lead0.station == code]
        vals = {
            f"{tag}_r{r}": 100 * np.isfinite(s[f"{tag}_prob_r{r}"]).mean()
            for tag in ("pn", "pda") for r in BOX_RADII
        }
        flag = ""
        if vals["pn_r1"] < 5 or vals["pda_r1"] < 5:
            flag = "  <-- 3x3 fails"
        print(
            f"  {code:4s} {meta['name']:30s} {vals['pn_r1']:6.1f}% {vals['pn_r2']:6.1f}% "
            f"{vals['pda_r1']:7.1f}% {vals['pda_r2']:7.1f}%{flag}"
        )
    return df


if __name__ == "__main__":
    main()
