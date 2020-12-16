import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append("../sir/")
import sir_discrete
import sir_ode
import sir_pde
import sirs_ode
import sirs_pde

# SIRS ODE model
# This simulation sets k = 1 (1 recovery per day), and varies the values of
# b and m. We simulate dynamics of the SIRS model for 0 <= t <= 500. We are
# interested in mapping out the relationship between (b,m) and whether the disease 
# is eradicated (i(t) = 0 at some t) or endemic (i(t) > 0 for all t). The phase transition
# diagram is the central object of interest.

# discretization = 500
# k = 1
# i0 = 0.0001
# bs = np.linspace(2, 10, discretization)
# ms = np.linspace(0.001, 7, discretization)
# bv, mv = np.meshgrid(bs, ms)

# tsteps = 50
# tend = 100
# teval = np.linspace(0, tend, tsteps)
# reinf = sirs_ode.SIRS_ODE(k, 0.1, 0.0001, i0)

# max_infection = np.zeros((discretization, discretization))
# min_infection = np.zeros((discretization, discretization))
# tend_infection = np.zeros((discretization, discretization))

# for a in range(discretization):
#     for b in range(discretization):
#         reinf.set_param(k=k,b=bv[a,b],m=mv[a,b])
#         (sol, obj) = reinf.simulate(teval)
#         max_infection[a,b] = np.max(sol.y[1])
#         min_infection[a,b] = np.min(sol.y[1])
#         tend_infection[a,b] = sol.y[1,-1]

# np.save("./sirs_simul_results/max_infection_fast2", max_infection)
# np.save("./sirs_simul_results/min_infection_fast2", min_infection)
# np.save("./sirs_simul_results/tend_infection_fast2", tend_infection)


### Plotting 
# Set parameters used for initialization
discretization = 500
k = 1
i0 = 0.0001
bs = np.linspace(2, 10, discretization)
ms = np.linspace(0.001, 7, discretization)
bv, mv = np.meshgrid(bs, ms)

# Load numpy arrays that were saved to disk
max_infection = np.load("./sirs_simul_results/max_infection_fast2.npy")
min_infection = np.load("./sirs_simul_results/min_infection_fast2.npy")
tend_infection = np.load("./sirs_simul_results/tend_infection_fast2.npy")

# Plot figures
plt.figure(figsize=(9, 8))
plt.contourf(ms, bs, max_infection)
plt.xlabel("m", fontsize=32)
plt.ylabel("b", fontsize=32)
plt.title("Peak Infection", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.savefig("../doc/final/sirs_variation/sirs_ode_peak_infection.png")

plt.figure(figsize=(9, 8))
plt.contourf(ms, bs, tend_infection)
plt.xlabel("m", fontsize=32)
plt.ylabel("b", fontsize=32)
plt.title("Steady State Infection", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.savefig("../doc/final/sirs_variation/sirs_ode_infection_steadystate.png")
