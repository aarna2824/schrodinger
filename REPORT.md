# Numerical Solutions of the Schrödinger Equation: Bound States, Wave Packets, and Tunneling

*Aarna Gattani · University of Edinburgh · 21st October 2026*

## Abstract

This project solves the one-dimensional Schrödinger equation numerically in Python. For the time-independent equation, a finite-difference eigenvalue solver finds the energy levels and wavefunctions of four potentials: the infinite square well, the finite square well, the harmonic oscillator, and a symmetric double well. For the two potentials with known analytic solutions, the computed energies agree with theory to better than 0.004%, and halving the grid spacing reduces the error by a factor of four, as expected for a second-order method. For the time-dependent equation, a split-step Fourier method evolves Gaussian wave packets. A packet released off-centre in a harmonic well oscillates with the classical period and holds its shape, as a coherent state should, and its mean position follows the classical trajectory to within $10^{-4}$. A packet with energy below a rectangular barrier tunnels through it, with a transmitted fraction of 18.5% compared with 19.2% from the plane-wave formula. Total probability is conserved to about $10^{-13}$.

## 1. Introduction

The Schrödinger equation describes how the quantum state of a particle evolves, and it can be solved exactly for only a handful of potentials. Numerical methods go beyond those cases, for example to a double well, and they also let us watch quantum dynamics directly: a wave packet oscillating in a well, or leaking through a barrier.

The goals of this project were:

1. Build a solver for the time-independent Schrödinger equation and validate it against analytic results.
2. Build a solver for the time-dependent equation and validate it against quantum and classical expectations.
3. Use the solvers on a system with no closed-form solution and to illustrate quantum tunnelling.

The allowed energy states of a quantum system are eigenfunctions of its Hamiltonian, in the same way that the resonant patterns of a vibrating plate are eigenmodes of its wave operator. Both are "which standing waves fit in this region?" problems.

## 2. Theory

### 2.1 Equations and units

The time-dependent Schrödinger equation in one dimension is

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \psi}{\partial x^2} + V(x)\,\psi,$$

and for states of definite energy $E$ it reduces to the time-independent equation

$$-\frac{\hbar^2}{2m}\psi''(x) + V(x)\,\psi(x) = E\,\psi(x).$$

All calculations use natural units with $\hbar = m = 1$, so energies, lengths and times are plain dimensionless numbers. Physical values are recovered by rescaling with a characteristic length and energy of the system being modelled.

### 2.2 Potentials studied

| Potential | Definition | Known result |
|---|---|---|
| Infinite square well | $V=0$ for $0<x<L$, walls impenetrable, $L=1$ | $E_n = n^2\pi^2/(2L^2)$, $n=1,2,\dots$ |
| Finite square well | $V=0$ for $\lvert x\rvert<1$, $V=50$ otherwise | Finite number of bound states, wave leaks into walls |
| Harmonic oscillator | $V=\tfrac12\omega^2x^2$, $\omega=1$ | $E_n = \omega(n+\tfrac12)$ |
| Double well | $V=(x^2-1.5^2)^2$ | None; solved numerically only |

### 2.3 Tunneling through a rectangular barrier

For a plane wave of energy $E$ hitting a barrier of height $V_0>E$ and width $w$, the transmission probability is

$$T = \frac{1}{1+\dfrac{V_0^2\sinh^2(\kappa w)}{4E(V_0-E)}},\qquad \kappa=\sqrt{2(V_0-E)}.$$

Classically $T$ would be zero. Quantum mechanically it is non-zero and falls off roughly exponentially with barrier width.

## 3. Methods

### 3.1 Time-independent solver (finite differences)

The position axis is divided into a uniform grid with spacing $\Delta x$. The second derivative is approximated by

$$\psi''_i \approx \frac{\psi_{i+1}-2\psi_i+\psi_{i-1}}{\Delta x^2}.$$

Substituting this into $-\tfrac12\psi''+V\psi=E\psi$ turns the equation into a matrix eigenvalue problem $H\psi=E\psi$, where $H$ is tridiagonal with diagonal entries $1/\Delta x^2+V_i$ and neighbouring entries $-1/(2\Delta x^2)$. The eigenvalues of $H$ are the energies and its eigenvectors are the wavefunctions. They are found with `numpy.linalg.eigh` and each eigenvector is normalized so that $\int|\psi|^2dx=1$. The wavefunction is forced to zero at the ends of the grid.

### 3.2 Time-dependent solver (split-step Fourier)

Over one small time step $\Delta t$ the evolution is split into potential and kinetic parts:

$$\psi(t+\Delta t)\approx e^{-iV\Delta t/2}\;\mathcal{F}^{-1}\!\left[e^{-ik^2\Delta t/2}\;\mathcal{F}\!\left[e^{-iV\Delta t/2}\psi(t)\right]\right],$$

where $\mathcal{F}$ is the fast Fourier transform. The potential term is a simple multiplication in position space, and the kinetic term is a simple multiplication in momentum space, so each factor is applied where it is easiest. Every factor is a pure phase, so the scheme conserves total probability exactly (up to rounding). The splitting error is second order in $\Delta t$. The FFT assumes periodic boundaries, so the domain is chosen large enough that no probability reaches the edges.

The initial state is a Gaussian wave packet

$$\psi_0(x)\propto \exp\!\left[-\frac{(x-x_0)^2}{2\sigma^2}+ik_0x\right],$$

normalized to unit probability. Here $x_0$ is the starting position, $\sigma$ the width, and $k_0$ the mean momentum.

### 3.3 Parameter choices

None of the numbers below come from experimental data. They are **chosen** to make each test clean, and each choice is explained.

| Quantity | Value | Why |
|---|---|---|
| Infinite well width $L$ | 1 | Simplest case; energies scale as $1/L^2$ |
| Grid points | 1000 (infinite well), 2000 (others) | Enough for errors below 0.004% while the matrix solve stays fast |
| Finite well depth, half-width | 50, 1 | Deep enough that all six computed states are bound |
| Harmonic frequency $\omega$ | 1 | Gives $E_n = n+\tfrac12$ |
| Harmonic domain | $[-10,10]$ | State $n=5$ has turning points near $\pm3.3$, so the wavefunction has decayed well before the edge |
| Double well | $(x^2-2.25)^2$ | Minima at $\pm1.5$ and a central barrier of height $\approx5.06$, above the ground-state energy so the lowest levels pair up |
| Packet start $x_0$, width $\sigma$ (harmonic) | 5, 1 | $\sigma=1/\sqrt{\omega}$ is the ground-state width, which makes the packet a coherent state; $x_0=5$ gives visible motion |
| Harmonic grid | 1024 points on $[-15,15]$ | Packet stays within $\pm6$ |
| Time step $\Delta t$ | 0.005 | About 1250 steps per oscillation period ($2\pi$) |
| Run length (harmonic) | 3 periods | Long enough to confirm the motion repeats |
| Barrier $V_0$, width | 3, 1 | A barrier thin enough to give measurable transmission |
| Packet momentum $k_0$ | 2 | Gives $E=k_0^2/2=2<V_0$, so any transmission is true tunneling |
| Packet width $\sigma$ (tunneling) | 5 | Wide in space, so narrow in momentum (spread $\approx 1/2\sigma = 0.1$) and the energy is fairly sharp |
| Packet start (tunneling) | $x_0=-40$ | About $8\sigma$ from the barrier, so there is no initial overlap |
| Tunneling domain, run time | $[-100,100]$ with 4096 points, $t=40$ | Speed is 2, so the packet reaches the barrier near $t=20$ and finishes about 40 units away on each side, well inside the domain, so nothing wraps around |

### 3.4 Validation checks

- Numerical eigenvalues compared with exact formulas.
- A grid-refinement test of the eigenvalue error.
- Total probability $\int|\psi|^2dx$ monitored through time.
- Mean position $\langle x\rangle(t)$ compared with the classical trajectory $x_0\cos\omega t$.
- Packet width and $\langle E\rangle$ compared with coherent-state predictions.
- Transmission compared with the formula in Section 2.3.

## 4. Results

### 4.1 Energy levels and wavefunctions

![Energy levels and wavefunctions for four potentials](figures/tise_levels.png)

*Figure 1. Wavefunctions for the six lowest states in each potential, each drawn at its energy. Top left: infinite well. Top right: finite well. Bottom left: harmonic oscillator. Bottom right: double well.*

**Infinite square well**

| n | Numerical E | Exact E | Error (%) |
|---|---|---|---|
| 1 | 4.9348 | 4.9348 | 0.00008 |
| 2 | 19.7391 | 19.7392 | 0.00033 |
| 3 | 44.4129 | 44.4132 | 0.00074 |
| 4 | 78.9558 | 78.9568 | 0.00131 |
| 5 | 123.3675 | 123.3701 | 0.00205 |
| 6 | 177.6476 | 177.6529 | 0.00295 |

The error grows with $n$ because higher states oscillate faster and are resolved by fewer grid points per wavelength.

**Convergence.** Refining the grid shows the expected second-order behaviour: the error in the ground-state energy falls by close to a factor of four each time the number of points is doubled.

| Grid points | Infinite well, error in $E_1$ (%) | Harmonic, error in $E_2$ (%) |
|---|---|---|
| 250 | 0.00131 | 0.1050 |
| 500 | 0.00033 | 0.0261 |
| 1000 | 0.00008 | 0.0065 |
| 2000 | 0.00002 | 0.0016 |

**Finite square well.** For depth 50 and half-width 1, the six lowest states all lie below the well depth and are bound: $E\approx 1.0143, 4.0492, 9.0791, 16.0526, 24.8708, 35.3181$. They are lower than the energies of an infinite well of the same width 2 ($1.23, 4.93, 11.10, \dots$) because the wavefunction leaks into the walls and so occupies a wider region. Instead of stopping at the edge, it decays exponentially in the classically forbidden region.

**Harmonic oscillator**

| n | Numerical E | Exact E | Error (%) |
|---|---|---|---|
| 0 | 0.50000 | 0.50000 | 0.00063 |
| 1 | 1.49998 | 1.50000 | 0.00104 |
| 2 | 2.49996 | 2.50000 | 0.00163 |
| 3 | 3.49992 | 3.50000 | 0.00223 |
| 4 | 4.49987 | 4.50000 | 0.00285 |
| 5 | 5.49981 | 5.50000 | 0.00347 |

The levels are evenly spaced by $\hbar\omega$ and the zero-point energy $\tfrac12\hbar\omega$ is reproduced. The even spacing is special to the harmonic oscillator. In the infinite well the gaps grow with $n$.

**Double well.** There is no analytic answer, so this case relies entirely on the numerics. The lowest energies are $1.97058,\ 2.01240,\ 4.90805,\ 5.77364,\ 7.83128,\ 9.94970$. The two lowest states form a nearly degenerate pair split by $\Delta E=0.04182$: the ground state is symmetric and the first excited state is antisymmetric. The small splitting is a signature of tunneling between the wells. A particle placed in one well would oscillate to the other with period $2\pi\hbar/\Delta E\approx150$ in these units. The second pair (4.91, 5.77) lies near the top of the central barrier ($V(0)\approx5.06$) and is split much more widely.

### 4.2 Coherent state in a harmonic well

![Wave packet in a harmonic well](figures/harmonic_packet.png)

*Figure 2. Left: quantum average position against the classical trajectory. Right: probability density over time.*

| Quantity | Result | Expectation |
|---|---|---|
| Max deviation of $\langle x\rangle(t)$ from $x_0\cos\omega t$ | $9.0\times10^{-5}$ | 0 |
| Maximum drift in total probability | $4.6\times10^{-13}$ | 0 |
| Position spread (std. dev. of $x$) | 0.7071 throughout | constant, $1/\sqrt2$ |
| Mean energy $\langle E\rangle$ | 13.0000 | $\tfrac12\omega+\tfrac12\omega^2x_0^2=13$ |

The packet's mean position follows the classical trajectory essentially exactly, which is Ehrenfest's theorem applied to a quadratic potential. The width does not change, so the packet never spreads, unlike a free packet. The energy matches the analytic value for a coherent state displaced to $x_0=5$.

### 4.3 Tunneling

![Tunneling through a barrier](figures/tunneling.png)

*Figure 3. Left: probability density at the start (blue) and end (orange) of the run, with the barrier in gray. Right: transmission against barrier width (log scale), simulation against the plane-wave formula.*

A packet of mean energy $E=2$ approached a barrier of height $V_0=3$ and width 1. At $t=40$:

| Quantity | Value |
|---|---|
| Transmitted probability | 0.1850 |
| Reflected probability | 0.8150 |
| Sum (conservation check) | 1.0000 |
| Plane-wave formula at $E=2$ | 0.1918 |

The mean energy is below the barrier top, yet about 18.5% of the probability is found on the far side. In the left panel the packet has split into a reflected part back near $x=-40$ and a transmitted part near $x=+40$. The area under each bump gives its probability.

The scan over barrier widths (right panel) shows transmission falling roughly exponentially:

| Barrier width | Simulation | Plane-wave formula |
|---|---|---|
| 0.2 | 0.875 | 0.915 |
| 0.6 | 0.458 | 0.494 |
| 1.0 | 0.185 | 0.192 |
| 1.4 | 0.0698 | 0.0658 |
| 1.8 | 0.0268 | 0.0217 |
| 2.2 | 0.0108 | 0.0070 |
| 2.6 | 0.0048 | 0.0023 |
| 3.0 | 0.0025 | 0.0007 |

The simulation agrees with the formula to within about 7% for widths up to 1.4, lies slightly below it for thin barriers, and lies above it for thick barriers, by a factor of about 3.4 at width 3. This is expected. The formula assumes a single energy, but the packet contains a spread of energies. Components with higher energy tunnel far more easily, and for thick barriers they dominate the transmitted probability. A single energy therefore underestimates what a real packet transmits.

## 5. Discussion

**Accuracy.** All eigenvalue errors were below 0.004% on the grids used, and the convergence test confirms the second-order scaling expected of the three-point derivative. The remaining error is from the finite grid, not from any mistake in the physics.

**Conservation laws.** Probability was conserved to about $10^{-13}$ in the time-dependent runs, consistent with the unitary structure of the split-step method. The agreement of $\langle E\rangle$ with its analytic value shows that energy is also handled correctly.

**Physical lessons.** The harmonic oscillator is unique in having evenly spaced levels and non-spreading coherent states. Confinement by a finite wall lowers energies compared with an infinite wall of the same width. Tunneling is a direct consequence of the wave nature of the solution, with exponential sensitivity to barrier width, and for a realistic packet the energy spread matters.

**Limitations.**

- The calculations are one-dimensional and use $\hbar=m=1$, so quantitative comparison with real systems requires rescaling.
- The FFT method assumes periodic boundaries, so the domain must be large enough to avoid wrap-around.
- The comparison with the tunneling formula is only approximate, because the formula assumes a single energy.
- The matrix solver builds a dense matrix, which is simple but slow for very large grids. A sparse or tridiagonal solver would scale better.

## 6. Conclusion

A compact numerical toolkit was written and validated for the one-dimensional Schrödinger equation. The time-independent solver reproduces analytic energies to better than 0.004%, converges at the expected second-order rate, and gives physically sensible results where no analytic answer exists. The time-dependent solver conserves probability to machine-level accuracy, reproduces the classical motion of a coherent state, and captures quantum tunneling, including the way a packet's energy spread changes the transmission for thick barriers.

## 7. Possible extensions

- Anharmonic oscillators (adding an $x^4$ term) compared with perturbation theory.
- Periodic potentials, to see energy bands emerge from a chain of wells and connect to solid-state physics.
- Rescaling to physical units, for example an electron in a nanometre-wide well.
- Superpositions of two eigenstates, to observe quantum beats at frequency $\Delta E/\hbar$.
- A two-dimensional version, which parallels the vibrational modes of a Chladni plate.

## References

1. D. J. Griffiths and D. F. Schroeter, *Introduction to Quantum Mechanics*, 3rd ed., Cambridge University Press (2018).
2. M. D. Feit, J. A. Fleck Jr. and A. Steiger, "Solution of the Schrödinger equation by a spectral method," *J. Comput. Phys.* 47, 412 (1982).
3. NumPy documentation: `numpy.linalg.eigh` and `numpy.fft`.
