import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams['font.size'] = 16
IT_pts, tol = 100, 1e-12

# --- Rashkov model ---
def rashkov_ode(t, y, params):
    CP, CPP, C_K, CP_K, K_I, C_K_I, CP_K_I, CPP_P, CP_P = y
    p = params
    PT, KT, ST, IT = p['PT'], p['KT'], p['ST'], p['IT']
    C = ST - CP - CPP - C_K - C_K_I - CP_K - CP_K_I - CP_P - CPP_P
    K = KT - K_I - C_K - C_K_I - CP_K - CP_K_I
    P = PT - CP_P - CPP_P   
    I = IT - C_K_I - CP_K_I - K_I
    dCP_dt = p['k2'] * C_K - p['k3'] * CP * K + p['k_7'] * CP_P - p['k7'] * CP * P + p['k_3'] * CP_K + p['k6'] * CPP_P + p['e4'] * CP_K_I - p['e_4'] * CP * K_I
    dCPP_dt = p['k4'] * CP_K + p['k_5'] * CPP_P - p['k5'] * CPP * P
    dC_K_dt = p['k1'] * C * K + p['e_1'] * C_K_I - (p['k_1'] + p['k2'] + p['e1'] * I) * C_K
    dCP_K_dt = p['k3'] * CP * K + p['e_3'] * CP_K_I - (p['k_3'] + p['k4'] + p['e3'] * I) * CP_K
    dK_I_dt = p['df'] * K * I + p['e2'] * C_K_I + p['e4'] * CP_K_I - p['dr'] * K_I - p['e_2'] * C * K_I - p['e_4'] * CP * K_I
    dC_K_I_dt = p['e1'] * C_K * I + p['e_2'] * C * K_I - (p['e_1'] + p['e2']) * C_K_I
    dCP_K_I_dt = p['e3'] * CP_K * I + p['e_4'] * CP * K_I - (p['e_3'] + p['e4']) * CP_K_I
    dCPP_P_dt = p['k5'] * CPP * P - (p['k_5'] + p['k6']) * CPP_P
    dCP_P_dt = p['k7'] * CP * P - (p['k_7'] + p['k8']) * CP_P
    return [dCP_dt, dCPP_dt, dC_K_dt, dCP_K_dt, dK_I_dt, dC_K_I_dt, dCP_K_I_dt, dCPP_P_dt, dCP_P_dt]

def get_rashkov_ss(params, IT_range, steady_tol=1e-6, max_time=1e8, time_step=10.0):
    y0 = np.zeros(9)
    CPP_final = []

    for IT in IT_range:
        p = params.copy()
        p['IT'] = IT
        t0 = 0.0
        y_current = np.array(y0)
        while t0 < max_time:
            sol = solve_ivp(
                rashkov_ode, [t0, t0 + time_step], y_current,
                args=(p,),
                rtol=tol, atol=tol, method="LSODA"
            )
            y_new = sol.y[:, -1]
            dy = np.abs(y_new - y_current)
            if np.all(dy < steady_tol):
                break
            y_current = y_new
            t0 += time_step
        CPP_final.append(y_new[1])
    return np.array(CPP_final)

# --- Hormesis model ---
def hormesis_ode(t, y, params):
    C1, C2, C3, A = y
    p = params
    MT, IT = p['MT'], p['IT']
    I = IT - C2 - 2.0 * C3
    M = MT - 2.0 * (C1 + C2 + C3 + A)
    dC1dt = p['kf1'] * M**2 - (p['kr1'] + p['k2'] + p['kf3'] * I) * C1 + p['kr3'] * C2
    dC2dt = p['kf3'] * C1 * I - (p['kr3'] + p['k4'] + p['k5'] * I) * C2
    dC3dt = p['k5'] * C2 * I
    dAdt = p['k2'] * C1 + p['k4'] * C2
    return [dC1dt, dC2dt, dC3dt, dAdt]

def get_hormesis_ss(params, IT_range, steady_tol=1e-6, max_time=1e8, time_step=10.0):
    y0 = np.zeros(4)
    A_final = []

    for IT in IT_range:
        p = params.copy()
        p['IT'] = IT
        t0 = 0.0
        y_current = np.array(y0)
        while t0 < max_time:
            sol = solve_ivp(
                hormesis_ode, [t0, t0 + time_step], y_current,
                args=(p,),
                rtol=tol, atol=tol, method="LSODA"
            )
            y_new = sol.y[:, -1]
            dy = np.abs(y_new - y_current)
            if np.all(dy < steady_tol):
                break
            y_current = y_new
            t0 += time_step
        A_final.append(y_new[3])
    return np.array(A_final)

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel A: Rashkov Model ---
IT_range_rashkov = np.logspace(-6, 3, IT_pts)
base_rashkov_params = {
    'k1': 45., 'k_1': 1., 'k2': 0.55, 'k3': 45., 'k_3': 1.,
    'k4': 4., 'k5': 0.3, 'k_5': 1., 'k6': 4.8, 'k7': 3.2,
    'k_7': 1., 'k8': 17., 'df': 1., 'dr': 1000., 'e1': 0.5,
    'e_1': 1., 'e2': 200., 'e_2': 0., 'e3': 0.5, 'e_3': 1.,
    'e4': 200., 'e_4': 0., 'ST': 8.44, 'KT': 2.5, 'PT': 2.5
}

# Hormetic Case
params_hormesis = base_rashkov_params.copy()
cpp_hormesis = get_rashkov_ss(params_hormesis, IT_range_rashkov)
ax1.plot(IT_range_rashkov, cpp_hormesis, 'o-', label='Hormesis ($k_4 > k_1$)', color='black', lw=2.5)

# Monotonic Case
params_monotonic = base_rashkov_params.copy()
params_monotonic.update({'df': 1000., 'dr': 1.})
cpp_monotonic = get_rashkov_ss(params_monotonic, IT_range_rashkov)
ax1.plot(IT_range_rashkov, cpp_monotonic, 's--', label='Monotonic Inhibition ($k_4 < k_1$)', color='black', lw=2.5)

ax1.set_xlabel("Total Inhibitor ($I_T$) [nM]")
ax1.set_ylabel('Pathway Output (${CPP}_{ss}$) [nM]')
ax1.set_xscale('log')
ax1.legend(frameon=False)
ax1.set_xlim([1e-6, 1e3])
ax1.set_yticks(np.linspace(0,0.35,8))
ax1.grid(False)
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# --- Panel B: Aggregation Model ---
IT_range_hormesis = np.logspace(-6, 3, IT_pts)
base_hormesis_params = {'kf1': 1.11, 'kr1': 8.56, 'k2': 0.10, 'kf3': 5.39, 'kr3': 4.77, 'k4': 8.23, 'k5': 5.37, 'MT': 0.005}

# Base case
params1 = base_hormesis_params.copy()
A_1 = get_hormesis_ss(params1, IT_range_hormesis)
ax2.plot(IT_range_hormesis, A_1, label='Base case', linestyle='solid', color='black', lw=2.5)

# Higher k2
params2 = base_hormesis_params.copy(); params2['k2'] = 1.0
A_2 = get_hormesis_ss(params2, IT_range_hormesis)
ax2.plot(IT_range_hormesis, A_2, label='High $k_2$', linestyle='dashed', color='black', lw=2.5)

# Lower k4
params3 = base_hormesis_params.copy(); params3['k4'] = 1.0
A_3 = get_hormesis_ss(params3, IT_range_hormesis)
ax2.plot(IT_range_hormesis, A_3, label='Low $k_4$', linestyle='dotted', color='black', lw=2.5)

ax2.set_xlabel("Total Inhibitor ($I_T$) [nM]")
ax2.set_ylabel("Aggregate ($A$) [nM]")
ax2.set_xscale('log')
ax2.legend(frameon=False)
ax2.set_xlim([1e-6, 1e3])
ax2.set_yticks(np.linspace(0,0.0021,8))
ax2.grid(False)
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout(pad=2.0)
plt.savefig('figure4.pdf')