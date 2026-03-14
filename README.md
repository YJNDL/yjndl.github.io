# ECP v1.3

ECP is a workflow for screening exfoliable crystallographic planes from non-vdW bulk materials.

This repository mainly hosts the project website and downloadable scripts.  
Detailed method description, update history, and workflow explanation are available on the project page:

- Website: [https://yjndl.github.io](https://yjndl.github.io)

## Core Script

- `downloads/version_5.0.py`
  The main production script.
- `downloads/run_version_5_parallel.py`
  Optional parallel launcher for running many representative structures with `version_5.0.py`.
- `downloads/prepare_newvasp_jobs.sh`
  Helper script for preparing per-material job folders.
- `downloads/submit_all_newvasp.sh`
  Helper script for submitting prepared jobs in batch.

`version_5.0.py` is the core analysis script.  
`run_version_5_parallel.py` is an optional Python launcher for multi-material parallel runs.  
These are the two maintained Python entry points in the current workflow.  
The shell scripts are templates and should be modified to match your own machine, paths, and cluster environment.

## Dependencies

Recommended Python version:

- Python 3.9-3.12

Required packages:

- `numpy`
- `pandas`
- `matplotlib`
- `pymatgen`
- `ase`
- `openpyxl`
- `tqdm`

Example:

```bash
pip install numpy pandas matplotlib pymatgen ase openpyxl tqdm
```

## Usage

### 1. Local Run

`version_5.0.py` reads the built-in config block at the top of the script by default.

Edit the config section in `version_5.0.py`, then run:

```bash
python version_5.0.py
```

Typical config items:

- `poscar_folder`: folder containing `mp-*.vasp`
- `excel_folder`: folder containing XRD Excel files
- `step_size`: scan step size
- `scan_mode`: `auto`, `event`, or `fixed`
- `xrd_min_intensity`: intensity threshold before scanning

### 2. Generate XRD Tables Only

If needed, you can use the script in XRD generation mode:

```bash
python version_5.0.py generate-xrd --input-dir INPUT_DIR --output-dir XRD
```

### 3. Parallel Launcher for Representative Sets

If you want to run many representative structures in parallel from a prepared
`materials_ecp_representatives/` directory, use:

```bash
python run_version_5_parallel.py
```

Before running, edit the `USER_CONFIG` block in `run_version_5_parallel.py`.

Most important fields:

- `representative_root`: root folder containing representative `mp-*.vasp` files
- `version_script`: path to `version_5.0.py`
- `parallel_workers`: number of worker processes
- `run_name`: fixed run name for resume mode
- `resume`: continue an interrupted run

### 4. Server / Cluster Run

For HPC usage, each material can be placed in its own job folder together with:

- the target `.vasp` file
- `version_5.0.py`
- `newvasp.sh`
- optional `XRD/`

Then submit with:

```bash
sbatch newvasp.sh
```

## Main Output

For each retained surface, the workflow can generate:

- screened slab results in `after_deal/*.xlsx`
- per-surface folders such as `face_h_k_l/`
- the evaluated slab model `POSCAR_h_k_l.vasp`
- the exported 2D model `POSCAR_2D_h_k_l.vasp`

The 2D model is generated from the minimum-bonding-density cleavage position found during slab evaluation.

## More Information

This `README` only covers practical usage.

For method details, workflow explanation, update notes, and examples, see:

- [https://yjndl.github.io](https://yjndl.github.io)
