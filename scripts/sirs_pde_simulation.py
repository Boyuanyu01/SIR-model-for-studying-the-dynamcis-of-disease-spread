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


# # Here is a simple simulation to just check that our PDE model is in working order.
# # Parameters
# k = 1
# b = 2
# p = 25
# gamma = 0.5
# m = 0
# M = 200
# i0 = np.zeros((M,M))
# i0[M//2, M//2] = 1

# pde0 = sirs_pde.SIRS_PDE_HETEROGENEOUS(k=k, b=b, m=m, ps=p, pi=gamma*p, pr=p, i0=i0, M=M)
# teval = np.linspace(0, 1000, 100)
# (sol, animi, anims, animr) = pde0.simulate(teval, figure=True)
# animi.save("infected_gamma50.gif", dpi=200, writer='imagemagick')
# anims.save("susceptible_gamma50.gif", dpi=200, writer='imagemagick')
# animr.save("recovered_gamma50.gif", dpi=200, writer='imagemagick')



# SIRS Heterogenous PDE model
# This simulation sets k = 1 (1 recovery per day), ps = pr = p, and pi = gamma*p, 
# where ps, pr, pi are the diffusion weights for susceptible, infected and recovered
# individuals. We simulate dynamics of the SIRS model for 0 <= t <= 1000. We are
# interested in mapping out the relationship between (b, gamma) and the speed of
# infection propagation. Note that the parameter b can be interpreted as the
# "social interaction" parameter governing infection propensity at each spatial location.
# In contrast, the diffusion weight describes how infected individuals move to different locations.
# On a macro-scale, diffusion can be interpreted as "travel"; lower diffusion weight correspond to 
# travel restrictions, whereas lower values of b correspond to lockdowns of community institutions.

# # Simulation parameters
# discretization = 10
# M = 200
# k = 1
# p = 25
# m = 4
# i0 = np.zeros((M,M))
# i0[M//2, M//2] = 1
# bs = np.linspace(2, 20, discretization)
# gammas = np.linspace(0.01, 1, discretization)
# bv, gv = np.meshgrid(bs, gammas)
# iprop_speed = np.zeros(shape=(discretization, discretization))
# rprop_speed = np.zeros(shape=(discretization, discretization))

# # Indices to extract s,i,r from solve_ivp solution
# Sind = np.arange(start=0, stop=M**2)
# Iind = np.arange(start=M**2,stop=(2*(M**2)))
# Rind = np.arange(start=(2*(M**2)),stop=(3*(M**2)))

# # helper matrix needed for calculating infection propagation speed
# xv, yv = np.meshgrid(np.arange(start=-M//2, stop=M//2), np.arange(start=-M//2, stop=M//2))
# norm_matrix = np.zeros(shape=(M,M))
# for x in range(xv.shape[0]):
#     for y in range(yv.shape[0]):
#         norm_matrix[x,y] = (xv[x,y]**2 + yv[x,y]**2)**(0.5)

# # Set up simulation
# tsteps = 100
# tend = 100
# teval = np.linspace(0, tend, tsteps)
# reinf = sirs_pde.SIRS_PDE_HETEROGENEOUS(k=k, b=bs[0], m=m, ps=p, pi = gammas[0]*p, pr=p, i0=i0, M=M)

# tol = 1e-4
# for a in range(discretization):
#     for b in range(discretization):
#         reinf.set_param(k=k,b=bv[a,b],m=m, ps=p, pi=gv[a,b]*p, pr=p)
#         (sol, animi, anims, animr) = reinf.simulate(teval, figure=False)
#         iprop_speed[a,b] = np.amax(np.multiply(norm_matrix, (sol.y[Iind,-1] > tol).reshape((M, M))).flatten())/((sol.y).shape[1])
#         rprop_speed[a,b] = np.amax(np.multiply(norm_matrix, (sol.y[Rind, -1] > tol).reshape((M, M))).flatten())/((sol.y).shape[1])

#         if ((a%(10) == 0) and (b%(10) ==0)):
#             print("(a={0}, b={1})".format(a, b))
        

# np.save("./sirs_pde_simul_results/infection_prop_speed_long", iprop_speed)
# np.save("./sirs_pde_simul_results/recovered_prop_speed_long", rprop_speed)


## Plotting
infection_speed = np.load("./sirs_pde_simul_results/infection_prop_speed_long.npy")

# parameters
discretization = 10
M = 200
k = 1
p = 25
m = 4
i0 = np.zeros((M,M))
i0[M//2, M//2] = 1
bs = np.linspace(2, 20, discretization)
gammas = np.linspace(0.01, 1, discretization)
bv, gv = np.meshgrid(bs, gammas)

plt.figure(figsize=(12,7))
plt.contourf(gammas, bs, infection_speed)
plt.xlabel("gamma", fontsize=24)
plt.ylabel("b", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Infection Propagation Speed", fontsize=24)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.savefig("../doc/final/infection_speed_propagation.png")
