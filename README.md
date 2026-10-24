# pressure_code

Pressure from planar velocity fields, for the PIV measurements and the DNS of
rotating Rayleigh–Bénard convection, by solving the pressure Poisson equation
with Neumann boundary conditions taken from the momentum equations.

The physics, grid and solvers are the same for both data sets; only the data
readers differ.

## Installation

Python 3.10 or newer, then

```
pip install -r requirements.txt
```

## Where the data are

The data are not part of this repository. `settings.py` looks for them in
`DATA_FOLDER`, which by default is the folder that **contains** this one:

```
DATA_FOLDER/
├── pressure_code/          this repository
├── PIV Data sets/          PIVlab .dat recordings   (run_piv.py, compare_solvers.py)
└── DNS_export/             DNS .npz exports          (run_dns.py, compare_solvers.py)
```

The DNS `.h5` snapshots are read from `DNS_DRIVE_FOLDER` (Google Drive).
Change these paths in `settings.py`, or the folder at the top of a run script,
if your data are elsewhere. Results are written next to the data, never into
this folder.

## How to run

Every `run_*.py` script has a **USER SETTINGS** block at the top. Change it,
then run the script from this folder:

```
cd pressure_code
python run_piv.py              # pressure + vorticity of a PIV recording
python run_dns.py              # DNS planes: full 3D and XY-only vs the DNS pressure
python run_dns_error_vs_z.py   # error against height, every plane of one DNS snapshot
python compare_solvers.py      # the four solvers on the same plane
```

## The files, in the order the data flows through them

| File | What it does |
|---|---|
| `settings.py` | All fixed numbers: Ra, Pr, PIV calibration, DNS grid, folders |
| `grid.py` | The grid of one direction: uniform (PIV) or tanh-stretched (DNS) |
| `load_piv.py` | Reads PIVlab `.dat` files into a "plane" dictionary |
| `load_dns.py` | Reads the DNS (`.npz` exports or `.h5` snapshots) into the same dictionary |
| `derivatives.py` | d/dx, d/dy, d²/dx², d²/dy² on either grid |
| `equations.py` | Momentum equations → dp/dx, dp/dy; the Poisson right-hand side (terms 1–5) |
| `poisson_setup.py` | Stencil weights, ghost cells, residual: shared by all solvers |
| `solvers/solver_splu.py` | Direct solver: sparse LU, factorised once per grid (default) |
| `solvers/solver_jacobi.py` | Jacobi iteration (vectorised) |
| `solvers/solver_gauss_seidel.py` | Gauss–Seidel iteration (vectorised, red–black ordering) |
| `solvers/solver_sor.py` | SOR iteration (vectorised, red–black ordering) |
| `solve_pressure.py` | `compute_pressure()`: equations + the solver chosen by name |
| `error_measures.py` | RMS error of a pressure field, as defined in the thesis |
| `plotting.py` | All figures |
| `run_*.py`, `compare_solvers.py` | The scripts you run |

## Things to know

* **Units.** Everything is non-dimensional: lengths in H, velocities in the
  free-fall velocity, time in H/U (see `settings.py`).
* **Arrays** are `f[i, j]` with `i` along x and `j` along y.
* **Sense of rotation.** The experiments rotate clockwise, the DNS
  anticlockwise. This sets the sign of the Coriolis term
  (`settings.CLOCKWISE` / `settings.ANTICLOCKWISE`) and must be right: at low
  Rossby number the Coriolis term is the largest term.
* **Term 4** (w and z-derivatives) is only available for the DNS. PIV is
  always solved "XY-only".
* **Margin.** The pressure is solved from `margin` points in from each edge
  (default 3, as in the original codes); see `poisson_setup.py`.
* **The constant.** The pressure is only defined up to a constant, so every
  solver returns it with its mean over the solved points removed.
* **Consecutive frames.** d/dt uses the fields one frame before and after.
  `run_piv.py` skips any field whose neighbours are not consecutive frames.

## Which solver?

All four solve the same equations and give the same pressure.
`splu` is exact and by far the fastest once the grid is factorised, so it is
the default. The iterative solvers are there to understand the method and to
cross-check; `compare_solvers.py` shows how many iterations each needs.
