import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

# Hormesis ODE system
def hormesis_ode(t, y, kf1, kr1, k2, kf3, kr3, k4, k5, IT):
    C1, C2, C3, A = y
    I = IT - C2 - 2.0 * C3
    M = MT - 2.0 * (C1 + C2 + C3 + A)
    dC1dt = kf1 * M**2 - (kr1 + k2 + kf3 * I) * C1 + kr3 * C2
    dC2dt = kf3 * C1 * I - (kr3 + k4 + k5 * I) * C2
    dC3dt = k5 * C2 * I
    dAdt = k2 * C1 + k4 * C2
    return [dC1dt, dC2dt, dC3dt, dAdt]

# Parameters
ITs = np.logspace(-6, 3, 1000)
tol = 1e-12
params = {
    'kf1': 1.11,
    'kr1': 8.56,
    'k2': 0.10,
    'kf3': 5.39,
    'kr3': 4.77,
    'k4': 8.23,
    'k5': 5.37,
    'MT': 0.005
}

# Simulation function
def simulate(params, steady_tol=1e-6, max_time=1e8, time_step=10.0):
    kf1, kr1, k2, kf3, kr3, k4, k5 = (
        params['kf1'], params['kr1'], params['k2'], params['kf3'],
        params['kr3'], params['k4'], params['k5']
    )
    global MT
    MT = params['MT']
    VA_final, A_final = [], []
    y0 = [0, 0, 0, 0]

    for IT in ITs:
        t0 = 0.0
        y_current = np.array(y0)
        while t0 < max_time:
            sol = solve_ivp(
                hormesis_ode, [t0, t0 + time_step], y_current,
                args=(kf1, kr1, k2, kf3, kr3, k4, k5, IT),
                rtol=tol, atol=tol, method="LSODA"
            )
            y_new = sol.y[:, -1]
            dy = np.abs(y_new - y_current)
            if np.all(dy < steady_tol):
                break
            y_current = y_new
            t0 += time_step
        C1, C2, C3, A = y_current
        A_final.append(A)
    return np.array(A_final)

# Run simulation
A_final = simulate(params)

# Compute baseline (IT = 0.0)
def simulate_baseline(params, steady_tol=1e-6, max_time=1e8, time_step=10.0):
    kf1, kr1, k2, kf3, kr3, k4, k5 = (
        params['kf1'], params['kr1'], params['k2'], params['kf3'],
        params['kr3'], params['k4'], params['k5']
    )
    MT = params['MT']
    VA_final, A_final = [], []
    y0 = [0, 0, 0, 0]
    IT = 0.0
    t0 = 0.0
    y_current = np.array(y0)
    while t0 < max_time:
        sol = solve_ivp(
            hormesis_ode, [t0, t0 + time_step], y_current,
            args=(kf1, kr1, k2, kf3, kr3, k4, k5, IT),
            rtol=tol, atol=tol, method="LSODA"
        )
        y_new = sol.y[:, -1]
        dy = np.abs(y_new - y_current)
        if np.all(dy < steady_tol):
            break
        y_current = y_new
        t0 += time_step
    C1, C2, C3, A = y_current
    A_final.append(A)
    return np.array(A_final)
A_baseline = simulate_baseline(params)

# Plot results
fig, axs = plt.subplots(1, 1, figsize=(9,5), sharex=True)
axs.semilogx(ITs, A_final, color='black')
axs.axhline(A_baseline, linestyle='--', color='black', label='Basal Level of Inhibition')
axs.set_xlabel("Total Inhibitor ($I_T$) [nM]")
axs.set_ylabel("Aggregate ($A$) [nM]")
axs.legend(frameon=False)
axs.grid(False)
axs.text(1e-4, max(A_final)*0.9, "Stimulation", color='black')
axs.text(30, A_baseline * 0.3, "Inhibition", color='black')
axs.set_xlim([1e-6, 1e3])
axs.set_yticks(np.linspace(0,0.00175,8))
plt.tight_layout()
plt.savefig('figure2.pdf')