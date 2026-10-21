"""
Schrodinger equation project (simple version)
Units: hbar = 1 and mass = 1, so everything is a plain number.

Run it with:   python3 schrodinger.py
Choose which parts to run with the three True/False switches below.
Close each plot window to move on to the next part.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)   # figures are saved here

RUN_PART_1 = True   # energy levels and wavefunctions
RUN_PART_2 = True   # wave packet in a harmonic well
RUN_PART_3 = True   # tunneling through a barrier


# =====================================================================
# PART 1: ENERGY LEVELS (time-independent Schrodinger equation)
# Idea: put x on a grid, turn the equation into a matrix, and let
# numpy find the eigenvalues (energies) and eigenvectors (wavefunctions).
# =====================================================================
def energy_levels(x, V, n_states):
    dx = x[1] - x[0]                      # spacing between grid points
    N = len(x)

    # Hamiltonian matrix  H = -1/2 d^2/dx^2 + V
    # The second derivative becomes  (psi[i+1] - 2 psi[i] + psi[i-1]) / dx^2
    main = 1.0 / dx**2 + V                # numbers on the diagonal
    side = -0.5 / dx**2 * np.ones(N - 1)  # numbers just next to the diagonal
    H = np.diag(main) + np.diag(side, 1) + np.diag(side, -1)

    energies, vectors = np.linalg.eigh(H)  # sorted lowest energy first

    E = energies[:n_states]
    psi = []
    for n in range(n_states):
        wave = vectors[:, n]
        wave = wave / np.sqrt(np.sum(wave**2) * dx)   # make total probability 1
        psi.append(wave)
    return E, psi


def draw_levels(ax, x, V, E, psi, title, xlim, ylim, size):
    """Draw the potential, then each wavefunction sitting at its energy."""
    ax.plot(x, V, "black", linewidth=2)
    for n in range(len(E)):
        ax.axhline(E[n], color="gray", linewidth=0.5, linestyle="--")
        ax.plot(x, E[n] + size * psi[n] / np.max(np.abs(psi[n])), label="n=" + str(n))
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("Energy")


if RUN_PART_1:
    n_states = 6
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ---- 1. Infinite square well (width L = 1) ----
    L = 1.0
    x = np.linspace(0, L, 1002)[1:-1]         # leave out the two wall points
    V = np.zeros(len(x))
    E, psi = energy_levels(x, V, n_states)
    print("INFINITE WELL")
    print("n   numerical   exact    percent error")
    for n in range(n_states):
        exact = ((n + 1) * np.pi) ** 2 / (2 * L**2)
        error = 100 * abs(E[n] - exact) / exact
        print(n + 1, "  ", round(E[n], 4), "  ", round(exact, 4), "  ", round(error, 5))
    draw_levels(axes[0, 0], x, V, E, psi, "Infinite square well",
                (0, L), (-12, E[-1] * 1.15), 8)

    # ---- 2. Finite square well (depth 50, half-width 1) ----
    x = np.linspace(-6, 6, 2000)
    V = np.where(np.abs(x) < 1.0, 0.0, 50.0)  # 0 inside, 50 outside
    E, psi = energy_levels(x, V, n_states)
    print("\nFINITE WELL (depth 50). Bound states have E below 50.")
    for n in range(n_states):
        print("n =", n, "  E =", round(E[n], 4))
    draw_levels(axes[0, 1], x, V, E, psi, "Finite square well",
                (-4, 4), (0, 55), 3)

    # ---- 3. Harmonic oscillator (omega = 1), exact E = n + 1/2 ----
    x = np.linspace(-10, 10, 2000)
    V = 0.5 * x**2
    E, psi = energy_levels(x, V, n_states)
    print("\nHARMONIC OSCILLATOR")
    print("n   numerical   exact    percent error")
    for n in range(n_states):
        exact = n + 0.5
        error = 100 * abs(E[n] - exact) / exact
        print(n, "  ", round(E[n], 5), "  ", exact, "  ", round(error, 5))
    draw_levels(axes[1, 0], x, V, E, psi, "Harmonic oscillator",
                (-6, 6), (0, 8), 0.5)

    # ---- 4. Double well (no exact answer exists) ----
    x = np.linspace(-4, 4, 2000)
    V = (x**2 - 1.5**2) ** 2
    E, psi = energy_levels(x, V, n_states)
    print("\nDOUBLE WELL (no exact answer)")
    for n in range(n_states):
        print("n =", n, "  E =", round(E[n], 5))
    print("Splitting of the lowest two levels =", round(E[1] - E[0], 5))
    draw_levels(axes[1, 1], x, V, E, psi, "Double well",
                (-3.2, 3.2), (0, 11), 0.5)

    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/tise_levels.png", dpi=150)
    print("\nSaved figures/tise_levels.png")
    plt.show()


# =====================================================================
# PART 2 AND 3: WAVE PACKETS MOVING IN TIME
# Idea: take many tiny time steps. Each step does three things:
#   1. a half-step of the potential (just multiply by a phase)
#   2. a full step of the kinetic energy (easy in momentum space,
#      so we use the FFT to jump there and back)
#   3. another half-step of the potential
# Multiplying by phases never changes the total probability.
# =====================================================================
def gaussian_packet(x, x0, sigma, k0):
    """A bell-shaped blob centered at x0, moving with momentum k0."""
    shape = np.exp(-(x - x0) ** 2 / (2 * sigma**2))
    wave = shape * np.exp(1j * k0 * x)
    wave = wave / np.sqrt(np.sum(np.abs(wave) ** 2) * (x[1] - x[0]))
    return wave


def time_step(psi, V, k, dt):
    """Move the wavefunction forward by one small time dt."""
    psi = np.exp(-0.5j * V * dt) * psi                     # potential, half step
    psi = np.fft.ifft(np.exp(-0.5j * k**2 * dt) * np.fft.fft(psi))   # kinetic
    psi = np.exp(-0.5j * V * dt) * psi                     # potential, half step
    return psi


if RUN_PART_2:
    # Harmonic well with omega = 1. Packet starts at x = 5, at rest.
    x = np.linspace(-15, 15, 1024)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)   # momentum values for the FFT
    V = 0.5 * x**2

    x0 = 5.0
    psi = gaussian_packet(x, x0, 1.0, 0.0)

    dt = 0.005
    n_steps = int(3 * 2 * np.pi / dt)              # three full oscillations

    times = []
    mean_x = []
    width = []
    norm = []
    snapshots = []

    for step in range(n_steps + 1):
        if step % 5 == 0:                          # record every 5th step
            prob = np.abs(psi) ** 2
            avg_x = np.sum(x * prob) * dx
            avg_x2 = np.sum(x**2 * prob) * dx
            times.append(step * dt)
            mean_x.append(avg_x)
            width.append(np.sqrt(avg_x2 - avg_x**2))
            norm.append(np.sum(prob) * dx)
            snapshots.append(prob)
        psi = time_step(psi, V, k, dt)

    times = np.array(times)
    mean_x = np.array(mean_x)
    classical = x0 * np.cos(times)                 # a ball on a spring

    # average energy = kinetic (from momentum space) + potential
    final = np.abs(np.fft.fft(psi)) ** 2
    kinetic = np.sum(0.5 * k**2 * final) / np.sum(final)
    potential = np.sum(V * np.abs(psi) ** 2) * dx

    print("\nHARMONIC WAVE PACKET")
    print("Biggest gap between <x> and classical:", np.max(np.abs(mean_x - classical)))
    print("Biggest change in total probability:", np.max(np.abs(np.array(norm) - 1)))
    print("Width stays between", round(min(width), 4), "and", round(max(width), 4))
    print("Average energy:", round(kinetic + potential, 4), " (expected 13)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(times, mean_x, label="quantum average position")
    axes[0].plot(times, classical, "--", label="classical ball on a spring")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("<x>")
    axes[0].legend()
    axes[1].imshow(np.array(snapshots).T, extent=[0, times[-1], x[0], x[-1]],
                   aspect="auto", origin="lower", cmap="magma")
    axes[1].set_ylim(-9, 9)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("x")
    axes[1].set_title("Probability density over time")
    plt.tight_layout()
    plt.savefig("figures/harmonic_packet.png", dpi=150)
    print("Saved figures/harmonic_packet.png")
    plt.show()


def barrier_transmission(E, V0, width):
    """Exact answer for a plane wave hitting a rectangular barrier (E < V0)."""
    kappa = np.sqrt(2 * (V0 - E))
    return 1 / (1 + V0**2 * np.sinh(kappa * width) ** 2 / (4 * E * (V0 - E)))


def run_tunnel(x, k, barrier_width, V0, psi_start, dt, n_steps):
    """Send the packet at a barrier. Return (reflected, transmitted)."""
    dx = x[1] - x[0]
    V = np.where(np.abs(x) < barrier_width / 2, V0, 0.0)
    psi = psi_start.copy()
    for step in range(n_steps):
        psi = time_step(psi, V, k, dt)
    prob = np.abs(psi) ** 2
    right = np.sum(prob[x > barrier_width / 2]) * dx   # got through
    left = np.sum(prob[x < -barrier_width / 2]) * dx   # bounced back
    return left, right, prob


if RUN_PART_3:
    x = np.linspace(-100, 100, 4096, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)

    V0 = 3.0            # barrier height
    width = 1.0         # barrier width
    k0 = 2.0            # packet momentum, so energy = k0^2 / 2 = 2 (below 3!)
    energy = k0**2 / 2
    psi_start = gaussian_packet(x, -40.0, 5.0, k0)

    dt = 0.01
    n_steps = 4000      # total time = 40

    reflected, transmitted, final_prob = run_tunnel(x, k, width, V0, psi_start, dt, n_steps)
    print("\nTUNNELING")
    print("Packet energy =", energy, "  barrier height =", V0)
    print("Transmitted (simulation):", round(transmitted, 4))
    print("Reflected   (simulation):", round(reflected, 4))
    print("Total:", round(transmitted + reflected, 4))
    print("Exact formula for one energy:", round(barrier_transmission(energy, V0, width), 4))

    # Repeat for different barrier widths
    widths = np.linspace(0.2, 3, 8)
    simulated = []
    formula = []
    for w in widths:
        r, t, p = run_tunnel(x, k, w, V0, psi_start, dt, n_steps)
        simulated.append(t)
        formula.append(barrier_transmission(energy, V0, w))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, np.abs(psi_start) ** 2, label="start (t = 0)")
    axes[0].plot(x, final_prob, label="end (t = 40)")
    axes[0].axvspan(-width / 2, width / 2, color="gray", alpha=0.5, label="barrier")
    axes[0].set_xlim(-60, 60)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("probability density")
    axes[0].legend()
    axes[1].semilogy(widths, simulated, "o", label="simulation")
    axes[1].semilogy(widths, formula, "-", label="exact formula")
    axes[1].set_xlabel("barrier width")
    axes[1].set_ylabel("transmission")
    axes[1].set_title("Thicker barrier, much less tunneling")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("figures/tunneling.png", dpi=150)
    print("Saved figures/tunneling.png")
    plt.show()
