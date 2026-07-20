from pathlib import Path
from typing import TypedDict

# paths
ROOT = Path(__file__).resolve().parents[1]
DATA_INPUT = ROOT / "data_input"
DATA_INTERIM = ROOT / "data_interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CALHABMAP_CSV = DATA_INPUT / "calhabmap.csv"

CHARM_FILES = {
    0: DATA_INPUT / "charmv3-0day.nc",
    1: DATA_INPUT / "charmv3-1day.nc",
    2: DATA_INPUT / "charmv3-2day.nc",
    3: DATA_INPUT / "charmv3-3day.nc",
}
LEADS = [0, 1, 2, 3]

# event thresholds for Pseudo nitzschia spp. (PN) (bloom) and particulate domoic acid (pDA) (toxicity)
PN_THRESHOLD_CELLS_L = 1e4 # cells / L
PDA_THRESHOLD_NG_ML = 0.5 # scaling 500 ng / L  --> ng / mL

# C-HARM variables
VAR_PN = "pseudo_nitzschia"
VAR_PDA = "particulate_domoic"

# calHABMAP columns
COL_PN_DELICATISSIMA = "pseudo_nitzschia_delicatissima_group_cells_l"
COL_PN_SERIATA = "pseudo_nitzschia_seriata_group_cells_l"
COL_PDA = "pda_ng_ml"

# stations
# calHABMAP reports a fixed lat/long coordinate for the stations --> take this as the median of the position we look at
STATIONS = {
    "TP":  dict(name="Trinidad Pier",                lat=41.0550, lon=-124.1470, has_pn=False, has_pda=True),
    "HUM": dict(name="Humboldt",                     lat=40.7783, lon=-124.1967, has_pn=False, has_pda=True),
    "HSB": dict(name="Humboldt South Bay",           lat=40.7103, lon=-124.2366, has_pn=False, has_pda=True),
    "BML": dict(name="Bodega Marine Lab",            lat=38.3161, lon=-123.0705, has_pn=True,  has_pda=False),
    "BBB": dict(name="Bodega Marine Lab Buoy",       lat=38.3126, lon=-123.0825, has_pn=True,  has_pda=False),
    "T00": dict(name="Tomales Bay Mouth",            lat=38.2308, lon=-122.9791, has_pn=True,  has_pda=False),
    "TBB": dict(name="Tomales Bay Mid-Channel Buoy", lat=38.1897, lon=-122.9286, has_pn=True,  has_pda=False),
    "T16": dict(name="Inner Tomales Bay",            lat=38.1180, lon=-122.8670, has_pn=True,  has_pda=False),
    "SCW": dict(name="Santa Cruz Wharf",             lat=36.9580, lon=-122.0170, has_pn=False, has_pda=True),
    "MW":  dict(name="Monterey Wharf",               lat=36.6037, lon=-121.8893, has_pn=True,  has_pda=False),
    "MBF": dict(name="Morro Bay Front Bay",          lat=35.3709, lon=-120.8587, has_pn=True,  has_pda=False),
    "MBB": dict(name="Morro Bay Back Bay",           lat=35.3300, lon=-120.8445, has_pn=True,  has_pda=False),
    "CPP": dict(name="Cal Poly Pier",                lat=35.1700, lon=-120.7410, has_pn=True,  has_pda=True),
    "SW":  dict(name="Stearns Wharf",                lat=34.4080, lon=-119.6850, has_pn=True,  has_pda=True),
    "SMP": dict(name="Santa Monica Pier",            lat=34.0080, lon=-118.4990, has_pn=True,  has_pda=True),
    "NBP": dict(name="Newport Beach Pier",           lat=33.6061, lon=-117.9311, has_pn=True,  has_pda=True),
    "SIO": dict(name="Scripps Pier",                 lat=32.8670, lon=-117.2570, has_pn=True,  has_pda=True),
}

# stations with both PN and pDA
BOTH_STATIONS = ["CPP", "SW", "SMP", "NBP", "SIO"]

# California-wide groups
CA_WIDE_PN = ["BML", "BBB", "T00", "TBB", "T16", "MW", "MBF", "MBB",
              "CPP", "SW", "SMP", "NBP", "SIO"]
CA_WIDE_PDA = ["TP", "HUM", "HSB", "SCW", "CPP", "SW", "SMP", "NBP", "SIO"]

# match-up extraction
    # 3x3 model grid point (Anderson et al. 2016)
    # T16 doesn't have a valid pixel in the 3x3 box
    # use a wider radius (5x5) to catch data
BOX_RADIUS = 1 # 3x3
BOX_RADIUS_WIDE = 2 # 5x5
BOX_RADII = [BOX_RADIUS, BOX_RADIUS_WIDE]

MODE_STRICT = "strict3x3" # primary
MODE_ADAPTIVE = "adaptive" # 3x3 when valid, else 5x5
MATCHUP_MODES = [MODE_STRICT, MODE_ADAPTIVE]
MIN_VALID_PIXELS = 1

# statistics
# two-sample komogorov-smirnov Dcrit = c(alpha) * sqrt((n1 + n2) / (n1 * n2))
# c(alpha) values based on chosen alpha (e.g. alpha = 0.05 --> 95% confidence level)
ALPHA = 0.05

KS_C_ALPHA = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63, 0.005: 1.73, 0.001: 1.95}

MAX_LAG_DAYS = 30

PREDICTION_POINTS_STEP = 0.01
