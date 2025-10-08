import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

# ============================================================================
# HORMESIS ODE SYSTEM
# ============================================================================
# This function implements the minimal mechanistic model of protein aggregation
# with inhibition described in the manuscript (Equations 3-6).
# The model demonstrates structural hormesis: low doses of inhibitor stimulate
# aggregation while high doses inhibit it.

def hormesis_ode(t, y, kf1, kr1, k2, kf3, kr3, k4, k5, IT):
    """
    Defines the ODE system for protein aggregation with inhibition.
    
    Parameters:
    -----------
    t : float
        Time variable (required by solve_ivp but not explicitly used)
    y : array-like
        State variables [C1, C2, C3, A]
        - C1: Initial complex formed from monomers (intermediate)
        - C2: Secondary productive complex (intermediate)
        - C3: Inactive off-pathway sink complex
        - A: Final aggregate concentration
    kf1 : float
        Forward rate constant for primary nucleation (M + M -> C1)
    kr1 : float
        Reverse rate constant for C1 dissociation
    k2 : float
        Rate constant for C1 -> A conversion
    kf3 : float
        Forward rate constant for inhibitor binding to C1
    kr3 : float
        Reverse rate constant for C2 dissociation
    k4 : float
        Rate constant for C2 -> A conversion
    k5 : float
        Rate constant for C3 formation (sequestration into off-pathway sink)
    IT : float
        Total inhibitor concentration [nM]
    
    Returns:
    --------
    list : [dC1dt, dC2dt, dC3dt, dAdt]
        Time derivatives of all species
    """
    C1, C2, C3, A = y
    
    # Calculate free inhibitor concentration using mass conservation (Equation 7)
    # The inhibitor is distributed among free form and bound forms in C2 and C3
    I = IT - C2 - 2.0 * C3  # C3 binds 2 inhibitor molecules
    
    # Calculate free monomer concentration using mass conservation
    # Each complex (C1, C2, C3, A) contains 2 monomers (dimeric)
    M = MT - 2.0 * (C1 + C2 + C3 + A)
    
    # Equation (3) from the manuscript: Rate of change of C1
    # Formation: kf1*M^2 (primary nucleation)
    # Consumption: (kr1 + k2)*C1 (dissociation and conversion to A)
    #              kf3*I*C1 (binding with inhibitor to form C2)
    # Formation from C2: kr3*C2 (reverse reaction)
    dC1dt = kf1 * M**2 - (kr1 + k2 + kf3 * I) * C1 + kr3 * C2
    
    # Equation (4) from the manuscript: Rate of change of C2
    # Formation: kf3*C1*I (C1 binds inhibitor)
    # Consumption: (kr3 + k4)*C2 (dissociation and conversion to A)
    #              k5*I*C2 (sequestration into C3 sink)
    dC2dt = kf3 * C1 * I - (kr3 + k4 + k5 * I) * C2
    
    # Equation (5) from the manuscript: Rate of change of C3 (off-pathway sink)
    # Formation only: k5*C2*I (irreversible sequestration)
    # This is the key step that causes inhibition at high inhibitor concentrations
    dC3dt = k5 * C2 * I
    
    # Equation (6) from the manuscript: Rate of change of aggregate A
    # Formation from both C1 and C2 (both are productive intermediates)
    dAdt = k2 * C1 + k4 * C2
    
    return [dC1dt, dC2dt, dC3dt, dAdt]

# ============================================================================
# PARAMETERS
# ============================================================================
# Range of total inhibitor concentrations to test (log-spaced from 10^-6 to 10^3 nM)
# This wide range is necessary to capture both the stimulation and inhibition regimes
ITs = np.logspace(-6, 3, 1000)

# Tolerance for numerical integration (high precision required for stiff ODEs)
tol = 1e-12

# Parameter values from Table 1 in the manuscript
# These values were chosen to demonstrate the hormetic response
params = {
    'kf1': 1.11,   # Primary nucleation rate [nM^-1 s^-1]
    'kr1': 8.56,   # C1 dissociation rate [s^-1]
    'k2': 0.10,    # C1 to A conversion rate [s^-1]
    'kf3': 5.39,   # Inhibitor binding to C1 [nM^-1 s^-1]
    'kr3': 4.77,   # C2 dissociation rate [s^-1]
    'k4': 8.23,    # C2 to A conversion rate [s^-1]
    'k5': 5.37,    # C3 formation (sequestration) rate [nM^-1 s^-1]
    'MT': 0.005    # Total monomer concentration [nM]
}

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================
def simulate(params, steady_tol=1e-6, max_time=1e8, time_step=10.0):
    """
    Simulates the hormesis model to steady state for multiple inhibitor concentrations.
    
    The function uses an adaptive approach: it integrates the ODEs in time steps
    and checks for convergence to steady state. This is necessary because the
    steady-state time varies significantly across different IT values.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all rate constants and total concentrations
    steady_tol : float
        Tolerance for declaring steady state (default: 1e-6)
        Steady state is reached when all species change by less than this value
    max_time : float
        Maximum simulation time [s] (default: 1e8)
    time_step : float
        Time step for checking convergence [s] (default: 10.0)
    
    Returns:
    --------
    np.array : A_final
        Array of steady-state aggregate concentrations for each IT value
    """
    # Extract parameters
    kf1, kr1, k2, kf3, kr3, k4, k5 = (
        params['kf1'], params['kr1'], params['k2'], params['kf3'],
        params['kr3'], params['k4'], params['k5']
    )
    
    # Set global MT (used in hormesis_ode function)
    global MT
    MT = params['MT']
    
    # Initialize storage for final aggregate concentrations
    A_final = []
    
    # Initial conditions: all species start at zero (no complexes or aggregates)
    y0 = [0, 0, 0, 0]

    # Loop through each inhibitor concentration
    for IT in ITs:
        t0 = 0.0  # Start time
        y_current = np.array(y0)  # Current state
        
        # Integrate until steady state is reached or max_time is exceeded
        while t0 < max_time:
            # Integrate ODEs over one time step using LSODA (adaptive stiff solver)
            # LSODA automatically switches between non-stiff and stiff methods
            sol = solve_ivp(
                hormesis_ode, [t0, t0 + time_step], y_current,
                args=(kf1, kr1, k2, kf3, kr3, k4, k5, IT),
                rtol=tol, atol=tol, method="LSODA"
            )
            
            # Extract final state from this time step
            y_new = sol.y[:, -1]
            
            # Check for convergence: if all species change by less than steady_tol,
            # we've reached steady state
            dy = np.abs(y_new - y_current)
            if np.all(dy < steady_tol):
                break
            
            # Update current state and time
            y_current = y_new
            t0 += time_step
        
        # Extract aggregate concentration at steady state
        C1, C2, C3, A = y_current
        A_final.append(A)
    
    return np.array(A_final)

# ============================================================================
# RUN MAIN SIMULATION
# ============================================================================
# Compute steady-state aggregate concentration for all IT values
A_final = simulate(params)

# ============================================================================
# COMPUTE BASELINE (IT = 0)
# ============================================================================
# This calculates the "basal level" of aggregation without any inhibitor
# This serves as the reference for identifying stimulation vs. inhibition
def simulate_baseline(params, steady_tol=1e-6, max_time=1e8, time_step=10.0):
    """
    Simulates the baseline aggregation level with no inhibitor (IT = 0).
    
    This represents the natural aggregation propensity of the system
    and serves as the reference for the hormetic response.
    """
    kf1, kr1, k2, kf3, kr3, k4, k5 = (
        params['kf1'], params['kr1'], params['k2'], params['kf3'],
        params['kr3'], params['k4'], params['k5']
    )
    MT = params['MT']
    A_final = []
    y0 = [0, 0, 0, 0]
    IT = 0.0  # No inhibitor
    
    t0 = 0.0
    y_current = np.array(y0)
    
    # Same integration strategy as main simulation
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

# ============================================================================
# PLOTTING - FIGURE 2
# ============================================================================
# This figure demonstrates the key result: hormetic dose-response
# As described in Section 3.1 of the manuscript, the curve shows:
# 1. Stimulation at low IT (A exceeds baseline)
# 2. Inhibition at high IT (A falls below baseline)
# 3. Peak at intermediate IT (~10^-4 nM, with ~2.5-fold increase)

fig, axs = plt.subplots(1, 1, figsize=(9,5), sharex=True)

# Plot aggregate concentration vs. inhibitor concentration (semi-log)
axs.semilogx(ITs, A_final, color='black')

# Add horizontal line showing basal aggregation level (IT = 0)
axs.axhline(A_baseline, linestyle='--', color='black', label='Basal Level of Inhibition')

# Labels and formatting
axs.set_xlabel("Total Inhibitor ($I_T$) [nM]")
axs.set_ylabel("Aggregate ($A$) [nM]")
axs.legend(frameon=False)
axs.grid(False)

# Add text annotations to identify stimulation and inhibition regions
axs.text(1e-4, max(A_final)*0.9, "Stimulation", color='black')
axs.text(30, A_baseline * 0.3, "Inhibition", color='black')

# Set axis limits and ticks
axs.set_xlim([1e-6, 1e3])
axs.set_yticks(np.linspace(0,0.00175,8))

plt.tight_layout()
plt.savefig('figure2.pdf')