"""Central configuration. Edit DATA_DIR if you want the data somewhere else.

Raw/processed data is kept outside the project folder (it is many GB; keep it out of synced or version-controlled folders).
Results (small CSV/PNG/JSON files) go to ../results inside the project folder.
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
RESULTS_DIR = Path(os.environ.get("GLU_RESULTS_DIR", PROJECT_DIR / "results"))

if os.name == "nt":
    _default_data = Path(r"C:\glucose_data")
else:
    _default_data = Path.home() / "glucose_data"
DATA_DIR = Path(os.environ.get("GLU_DATA_DIR", _default_data))

RAW_DIR = DATA_DIR / "vitaldb_raw"          # one .npz per case (windows around glucose readings)
PROC_DIR = DATA_DIR / "processed"           # beats / segments / reading table
MUST_DIR = DATA_DIR / "must"                # MUST dataset (manual download from Mendeley)

# --- signal settings (same as the original paper) ---
PPG_TRACK = "SNUADC/PLETH"
FS_RAW = 500            # VitalDB SNUADC waveform rate
FS = 100                # analysis rate
HALF_WINDOW_S = 8 * 60  # +/- 8 min around each glucose reading (paper)
BAND = (0.5, 8.0)       # Butterworth band-pass (paper)
FILTER_ORDER = 3
SEG10_S = 10            # 10-s segments (paper baseline)
BEAT_LEN = 100          # 1-s beat window at 100 Hz (paper)
COS_SIM_MIN = 0.85      # template-matching threshold (paper)

# caps per glucose reading (keeps memory bounded; readings are the unit of evaluation)
MAX_SEG10_PER_READING = 24
MAX_BEATS_PER_READING = 120

# plausibility filters on glucose labels (mg/dL)
GLU_MIN, GLU_MAX = 20, 600

SEED = 42
N_FOLDS = 5

for d in (RESULTS_DIR,):
    d.mkdir(parents=True, exist_ok=True)
