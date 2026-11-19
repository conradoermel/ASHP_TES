# Installation and first run

## 1. Install

**With conda (recommended).** From the repository folder:

```bash
conda env create -f environment.yml
conda activate ashp_tes
```

This installs Python 3.12 and the required packages (numpy, scipy,
matplotlib, pandas, xarray, netCDF4, tqdm, CoolProp) from conda-forge.


**With pip.** Python 3.10–3.12:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```


## 2. Run

```bash
python Main_program.py
```

A full year takes a while: the heat pump model solves the refrigerant cycle
with CoolProp for each of the 8760 hours, about 0.3 s per simulated hour on a laptop.
Expect 40 minutes to an hour. The two `tqdm` progress bars show the thermal load
(fast) and the heat pump calculation (slow).

