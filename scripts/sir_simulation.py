import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append("../sir/")
import sir_discrete
import sir_ode
import sir_pde

# # Discrete model

# # Here is a simple simulation. With these parameters, we are
# # simulating the setting where there is just 0.01% infected individuals at time 0,
# # 5% infected people recover per day, and each person interacts with 10.
# k = 0.05
# b = 10
# N = 10000
# t = 100
# s0 = 0.9999
# i0 = 0.0001
# s, i, r = sir_discrete.simulation(b, k, t, N, s0, i0)
# times = range(t + 1)
# plt.figure()
# plt.plot(times, s, c="blue")
# plt.plot(times, i, c="red")
# plt.plot(times, r, c="green")
# plt.title("SIR with k = {0}, b = {1}, N = {2}, s0 = {3}, i0 = {4}".format(k, b, N, s0, i0))
# plt.xlabel("Time")
# plt.ylabel("Population_ratio")
# plt.legend(("Susceptible", "Infected", "Recovered"))

# plt.savefig("../doc/checkpoint/sir_discrete_simulation0.png")

# # Here is a simple simulation. With these parameters, we are
# # simulating the setting where there is just 0.01% infected individuals at time 0,
# # 5% infected people recover per day, and each person interacts with 1.
# k = 0.05
# b = 1
# N = 10000
# t = 100
# s0 = 0.9999
# i0 = 0.0001
# s, i, r = sir_discrete.simulation(b, k, t, N, s0, i0)
# times = range(t + 1)
# plt.figure()
# plt.plot(times, s, c="blue")
# plt.plot(times, i, c="red")
# plt.plot(times, r, c="green")
# plt.title("SIR with k = {0}, b = {1}, N = {2}, s0 = {3}, i0 = {4}".format(k, b, N, s0, i0))
# plt.xlabel("Time")
# plt.ylabel("Population_ratio")
# plt.legend(("Susceptible", "Infected", "Recovered"))

# plt.savefig("../doc/checkpoint/sir_discrete_simulation1.png")

# # Phase diagram for s(t) = 0
# discretization = 50
# ks = np.linspace(0.01, 0.2, discretization)
# bs = np.linspace(0.1, 5, discretization)
# N = 10000
# s0 = 0.9999
# i0 = 0.0001
# t = 50

# kv, bv = np.meshgrid(ks, bs)
# times = np.zeros((discretization, discretization))
# for ii in range(discretization):
#     for jj in range(discretization):
#         s, i, r = sir_discrete.simulation(bv[ii, jj], kv[ii, jj], t, N, s0, i0)
#         checks, t0 = sir_discrete.checksifs(s, 0)
#         if checks == False:
#             times[ii,jj] = t
#         else:
#             times[ii,jj] = t0

# plt.figure()
# plt.contourf(ks, bs, times, vmin=0, vmax=t)
# plt.xlabel("k")
# plt.ylabel("b")
# plt.title("Time to full infection (s(t) = 0)")
# plt.colorbar()
# plt.savefig("../doc/checkpoint/phase_transition_full_infection_discrete.png")

# # Phase diagram for i(t) > 0.5
# discretization = 50
# ks = np.linspace(0.01, 0.2, discretization)
# bs = np.linspace(0.1, 5, discretization)
# N = 10000
# s0 = 0.9999
# i0 = 0.0001
# t = 50

# kv, bv = np.meshgrid(ks, bs)
# times = np.zeros((discretization, discretization))
# for ii in range(discretization):
#     for jj in range(discretization):
#         s, i, r = sir_discrete.simulation(bv[ii, jj], kv[ii, jj], t, N, s0, i0)
#         checki, t0 = sir_discrete.checksifi(i, 0.5)
#         if checki == False:
#             times[ii,jj] = t
#         else:
#             times[ii,jj] = t0

# plt.figure()
# plt.contourf(ks, bs, times, vmin=0, vmax=t)
# plt.xlabel("k")
# plt.ylabel("b")
# plt.title("Time to i(t) > 0.5")
# plt.colorbar()
# plt.savefig("../doc/checkpoint/phase_outnumber_discrete.png")

# # ODE model

# # Here is a simple simulation. With these parameters, we are
# # simulating the setting where there is just 0.1% infected individuals at time 0,
# # 10% infected people recover per day, and each person interacts with 10.
# k = 0.05
# b = 10
# N = 1
# i0 = 0.0001
# ode0 = sir_ode.SIR_ODE(k, b, i0, N)
# teval = np.linspace(0, 50, 1000)
# (sol, obj0) = ode0.simulate(teval, figure=True, normalized_plot=True)
# obj0.savefig("../doc/final/sir_ode_simulation0.png")


# # Here is another simulation where we suppose people engage in social distancing,
# # interacting with only one other person. 
# k = 0.05
# b = 1
# N = 1
# i0 = 0.0001
# ode1 = sir_ode.SIR_ODE(k, b, i0, N)
# teval = np.linspace(0, 50, 1000)
# (sol, obj1) = ode1.simulate(teval, figure=True, normalized_plot=True)
# obj1.savefig("../doc/final/sir_ode_simulation1.png")


# Let's investigate phase transitions in the dynamics

# First, we look at the time to full infection (i.e. smallest t such that s(t) = 0)
t_to_full = lambda t, sir : sir[0]*(sir[0] > 1e-5)
discretization = 50
ks = np.linspace(0.01, 0.2, discretization)
bs = np.linspace(0.1, 5, discretization)
N = 10000
i0 = 0.0001
teval = np.linspace(0, 50, 1000)

kv, bv = np.meshgrid(ks, bs)
times = np.zeros((discretization, discretization))
ode = sir_ode.SIR_ODE(0.01, 1, i0, N)
for i in range(discretization):
    for j in range(discretization):
        ode.set_param(kv[i,j], bv[i,j])
        (sol, obj) = ode.simulate(teval, normalized_plot=False, events=t_to_full)
        if sol.t_events[0].size == 0:
            times[i,j] = teval[-1]
        else:
            times[i,j] = sol.t_events[0][0]

plt.figure(figsize=(9, 8))
plt.contourf(ks, bs, times, vmin=0, vmax=teval[-1])
plt.xlabel("k", fontsize=32)
plt.ylabel("b", fontsize=32)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Time to full infection (s(t) = 0)", fontsize=26)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.savefig("../doc/final/phase_transition_full_infection.png")

# Second, we look at the time at which infectious people outnumber everyone else (i(t) > 0.5))
t_to_outnumber = lambda t, sir : (0.5 > sir[1])
discretization = 50
ks = np.linspace(0.01, 0.2, discretization)
bs = np.linspace(0.1, 3, discretization)
N = 10000
i0 = 0.0001
teval = np.linspace(0, 50, 1000)

kv, bv = np.meshgrid(ks, bs)
times = np.zeros((discretization, discretization))
ode = sir_ode.SIR_ODE(0.01, 1, i0, N)
for i in range(discretization):
    for j in range(discretization):
        ode.set_param(kv[i,j], bv[i,j])
        (sol, obj) = ode.simulate(teval, normalized_plot=False, events=t_to_outnumber)
        if sol.t_events[0].size == 0:
            times[i,j] = teval[-1]
        else:
            times[i,j] = sol.t_events[0][0]

plt.figure(figsize=(9, 8))
plt.contourf(ks, bs, times, vmin=0, vmax=teval[-1])
plt.xlabel("k", fontsize=32)
plt.ylabel("b", fontsize=32)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Time to i(t) > 0.5", fontsize=26)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.savefig("../doc/final/phase_transition_outnumber.png")




# # PDE Model 

# # Here is a simple simulation to just check that our PDE model is in working order.
# # Parameters
# k = 0.075
# b = 1
# p = 10
# M = 200
# i0 = np.zeros((M,M))
# s0 = np.zeros((M, M))
# i0[M//2, M//2] = 1

# pde0 = sir_pde.SIR_PDE(k, b, p, i0, M=M)
# teval = np.linspace(0, 500, 100)
# (sol, animi, anims, animr) = pde0.simulate(teval, figure=True)
# animi.save("../doc/final/pde_infected.gif", dpi=200, writer='imagemagick')
# anims.save("../doc/final/pde_susceptible.gif", dpi=200, writer='imagemagick')
# animr.save("../doc/final/pde_recovered.gif", dpi=200, writer='imagemagick')


# Here, we examine how different values of p correspond to different numbers of individuals who had been infected at any time point.
# We keep the values k = 0.075 and b = 1 as these are near the boundary of the phase transition as
# seen in our midterm checkpoint. We will simulate up to time 500 with 1000 time points.

# k = 0.075
# b = 1
# M = 200
# Sind = np.arange(start=0, stop=M**2)
# Iind = np.arange(start=M**2,stop=(2*(M**2)))
# Rind = np.arange(start=(2*(M**2)),stop=(3*(M**2)))
# ps = np.linspace(start=0.1, stop=50, num=100)
# fraction_infected = np.zeros(ps.shape[0])
# i0 = np.zeros((M,M))
# i0[M//2, M//2] = 1
# teval = np.linspace(0, 500, 1000)
# pde = sir_pde.SIR_PDE(k, b, 0.1, i0, M=M)
# for j in np.arange(ps.shape[0]):
#     pde.set_param(k, b, ps[j])
#     (sol, animi, anims, animr) = pde.simulate(teval, figure=False)

#     # Calculate total number of people who had been infected
#     fraction_infected[j] = 1 - np.sum(sol.y[Sind, teval.shape[0]-1])/(M**2)

# plt.plot(ps, fraction_infected, color="blue")
# plt.xlabel("p")
# plt.ylabel("Fraction who had once been infected")
# plt.title("Infection spread as p varies : k = {0}, b = {1}, 0 <= t <= {2}".format(k, b,teval[-1]))
# plt.savefig("../doc/final/infection_vs_p.png")


# Here, we examine the difference in the infection dynamics when the initial infection is in the corner of the grid versus when the
# initial infection is in the center of the grid. We imagine that the difference will be most striking when the diffusion of infection
# happens quite quickly. So we will select p = 25 using our results from Question 1. Note that p = 25 corresponds to the case when one
# infection at the center of the grid diffuses to cover roughly 50% of the grid by time point t = 500.

# k = 0.075
# b = 1
# p = 25
# M = 200

# i0center = np.zeros((M,M))
# i0center[M//2, M//2] = 1

# i0corner = np.zeros((M,M))
# i0corner[0,0] = 1

# teval = np.linspace(0, 500, 1000)
# pde_center = sir_pde.SIR_PDE(k, b, p, i0center, M=M)
# pde_corner = sir_pde.SIR_PDE(k, b, p, i0corner, M=M)

# teval = np.linspace(0, 500, 100)
# (solcenter, animicenter, animscenter, animrcenter) = pde_center.simulate(teval, figure=True)
# (solcorner, animicorner, animscorner, animrcorner) = pde_corner.simulate(teval, figure=True)

# animicenter.save("../doc/final/pde_center_infected.gif", dpi=200, writer='pillow')
# animscenter.save("../doc/final/pde_center_susceptible.gif", dpi=200, writer='pillow')
# animrcenter.save("../doc/final/pde_center_recovered.gif", dpi=200, writer='pillow')

# animicorner.save("../doc/final/pde_corner_infected.gif", dpi=200, writer='pillow')
# animscorner.save("../doc/final/pde_corner_susceptible.gif", dpi=200, writer='pillow')
# animrcorner.save("../doc/final/pde_corner_recovered.gif", dpi=200, writer='pillow')
