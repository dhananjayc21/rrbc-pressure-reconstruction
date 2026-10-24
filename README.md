# pressure_code

Pressure from planar velocity fields, for the PIV measurements and the DNS of
rotating Rayleigh–Bénard convection, by solving the pressure Poisson equation
with Neumann boundary conditions taken from the momentum equations.

The code follows the PIV notebook (`Pressure_Poisson_PIV.ipynb`): the same
variable names (`term1` … `term5`, `B`, `A_p`, `A_w`, `A_e`, `A_s`, `A_n`,
`Vor`, `Div_xy`), plain functions, `for` and `while` loops, and comments on
every step. The physics, grid and solvers are the same for PIV and DNS; only
the readers differ.

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
Change these paths in `settings.py` if your data are elsewhere.

## How to run

Every `run_*.py` script has a **SETTINGS** block at the top (the things to
change are marked with `---->`). Change it, then run the script from this folder:

```
cd pressure_code
python run_piv.py              # pressure + vorticity of a PIV recording
python run_dns.py              # DNS planes: full 3D and XY-only vs the DNS pressure
python run_dns_error_vs_z.py   # error against height, every plane of one DNS snapshot
python compare_solvers.py      # the four solvers on the same plane
```

## The files, in the order the data go through them

| File | What it does |
|---|---|
| `settings.py` | All fixed numbers: water properties, Ra, Pr, PIV calibration, DNS grid, folders |
| `grid.py` | The grid of one direction: uniform (PIV) or tanh-stretched (DNS) |
| `read_piv.py` | Reads PIVlab `.dat` files: velocity at t-1, t, t+1 (dimensionless) and the grid |
| `read_dns.py` | Reads the DNS (`.npz` exports or `.h5` snapshots): velocity, w and z-derivatives, DNS pressure |
| `derivatives.py` | `d_dx`, `d_dy`, `d2_dx2`, `d2_dy2` on either grid |
| `equations.py` | `dp_dx`, `dp_dy` (boundary conditions) and `B` = term1 + … + term5 |
| `coefficients.py` | `A_p`, `A_w`, `A_e`, `A_s`, `A_n`, the ghost cells, the residual |
| `poisson_matrix.py` | All equations as one matrix; making the data consistent |
| `solvers/direct.py` | Direct solver: sparse LU, factorised once per grid (default) |
| `solvers/jacobi.py` | Jacobi iteration, as in the notebook |
| `solvers/gauss_seidel.py` | Gauss–Seidel iteration (red–black) |
| `solvers/sor.py` | SOR iteration (red–black) |
| `solve_pressure.py` | `prepare_poisson()` (once per grid) and `compute_pressure()` (every snapshot) |
| `error_measures.py` | RMS error of a pressure field, as defined in the thesis |
| `plotting.py` | The figures |
| `run_*.py`, `compare_solvers.py` | The scripts you run |

## Things to know

* **Units.** Everything is dimensionless: lengths in H, velocities in
  `U_ndim = sqrt(g beta dT H)`, time in `T_ndim = H / U_ndim` (see `settings.py`).
* **Arrays** are `f[x][y]`: the first index along x, the second along y.
* **Sense of rotation.** The experiments rotate clockwise, the DNS
  anticlockwise. This sets the sign of the Coriolis terms
  (`settings.CLOCKWISE` / `settings.ANTICLOCKWISE`) and must be right: at low
  Rossby number the Coriolis term is the largest term.
* **Term 4** (w and z-derivatives) is only available for the DNS. PIV is
  always solved "XY-only".
* **Margin.** The pressure is solved at `X[3:-3, 3:-3]`, with the ghost points
  at index 2 and -3, as in the notebook (`settings.MARGIN = 3`).
* **The constant.** The pressure is only known up to a constant, so every
  solver returns it with its mean over the solved points removed.
* **Consistent data.** Before solving, the small uniform mismatch between `B`
  and the boundary values is removed (see `poisson_matrix.py`); without it
  the iterative solvers never converge.
* **Consecutive frames.** d/dt uses the fields one frame before and after;
  `run_piv.py` skips any field whose neighbours are not consecutive frames.

## Which solver?

All four solve the same equations and give the same pressure
(`compare_solvers.py`: they agree to about 10⁻⁶ % of the pressure range).
`direct` is exact and by far the fastest once the grid is factorised, so it is
the default. The iterative solvers are there to understand the method and to
cross-check: on the PIV grid SOR needs about 2 400 iterations, Gauss–Seidel
about 126 000 and Jacobi about 270 000.
