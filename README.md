# Numerical Solutions of the Schrödinger Equation

A small Python project that solves the one-dimensional Schrödinger equation numerically, in two parts:

1. **Energy levels and wavefunctions** (time-independent): infinite well, finite well, harmonic oscillator, and a double well.
2. **Wave packets in time** (time-dependent): a coherent state oscillating in a harmonic well, and a packet tunneling through a barrier.

Everything is in one commented script, `schrodinger.py`, written to be readable. Units are ħ = m = 1.

![Energy levels](figures/tise_levels.png)

## Key results

| Test | Result |
|---|---|
| Infinite-well and harmonic energies vs exact | Error below 0.004% |
| Grid refinement (double the points) | Error drops about 4×, as expected for a second-order method |
| Harmonic wave packet vs classical orbit | ⟨x⟩(t) matches x₀cos(ωt) to 1e-4 |
| Probability conservation | Drift about 1e-13 |
| Tunneling (E = 2, barrier height 3, width 1) | 18.5% transmitted (plane-wave formula: 19.2%) |

See [REPORT.md](REPORT.md) for the full write-up, including how each parameter was chosen.

![Harmonic packet](figures/harmonic_packet.png)
![Tunneling](figures/tunneling.png)

## How to run

You need Python 3 and three libraries:

```
pip install -r requirements.txt
python3 schrodinger.py
```

Plot windows open one after another; close each to continue. Figures are also saved to `figures/`. To run only part of the project, change the `RUN_PART_1`, `RUN_PART_2` and `RUN_PART_3` switches at the top of the script from `True` to `False`.

## Methods in brief

- **Energy levels:** the grid version of the equation is a matrix, and its eigenvalues and eigenvectors are the energies and wavefunctions (`numpy.linalg.eigh`).
- **Time evolution:** the split-step Fourier method. Each small time step applies half the potential, then the kinetic energy in momentum space using the FFT, then the other half of the potential. Because every step is a phase, total probability is conserved.

## Project structure

```
schrodinger-numerics/
├── schrodinger.py     # all the code
├── REPORT.md          # full report
├── requirements.txt
├── LICENSE
└── figures/           # generated plots
```

## Things to try

- Change `V0`, `width` or `k0` in the tunneling section and predict the result before running.
- Add an anharmonic term such as `+ 0.1 * x**4` to the harmonic potential.
- Change the packet width in the harmonic section so it is no longer a coherent state and watch it "breathe".

## License

MIT, see [LICENSE](LICENSE).
