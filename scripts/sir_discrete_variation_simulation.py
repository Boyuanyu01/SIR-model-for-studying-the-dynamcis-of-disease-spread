import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append("../sir/")
import sir_discrete
import sir_ode
import sir_pde
import sir_discrete_variation

# # Discrete model
'''
# # Here is a simple simulation. With these parameters, we are
# # simulating the setting where there is just 0.01% infected individuals at time 0,
# # 5% infected people recover per day, and each person interacts with 10.
k = 0.05
b = 10
N = 10000
t = 100
s0 = 0.9999
i0 = 0.0001
D0 = 1
T0 = 200
s, i, r = sir_discrete_variation.simulation(b, k, t, N, s0, i0, T0, D0)
times = range(t + 1)
plt.figure()
plt.plot(times, s, c="blue")
plt.plot(times, i, c="red")
plt.plot(times, r, c="green")
plt.title("SIR with D0 = L, T0 = 200")
plt.xlabel("Time")
plt.ylabel("Population_ratio")
plt.legend(("Susceptible", "Infected", "Recovered"))
'''
# # Here is a simple simulation. With these parameters, we are
# # simulating the setting where there is just 0.01% infected individuals at time 0,
# # 5% infected people recover per day, and each person interacts with 10.
# # one person interacts with people with a square of 2D0 = L/4
# # No 'lock down'
k = 0.05
b = 10
N = 10000
t = 100
s0 = 0.9999
i0 = 0.0001
D0 = 0.125
T0 = 200
T1 = 200
s, i, r = sir_discrete_variation.simulation(b, k, t, N, s0, i0, T0, D0, T1, True)
times = range(t + 1)
plt.figure()
plt.plot(times, s, c="blue")
plt.plot(times, i, c="red")
plt.plot(times, r, c="green")
plt.title("SIR with D0 = L/8, T0 = 200")
plt.xlabel("Time")
plt.ylabel("Population_ratio")
plt.legend(("Susceptible", "Infected", "Recovered"))
#plt.savefig('../doc/final/sir_discrete_grid15.png')
