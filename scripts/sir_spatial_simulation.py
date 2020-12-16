import matplotlib.pyplot as plt
import sys
from math import pi
sys.path.append("../sir/")
import sir_spatial_model

# # Discrete model with spacial structure

# # Here is a example simulation. With these parameters, we are
# # simulating the setting where there is just 0.01% infected individuals at time 0,
# # 5% infected people recover per day, and each person interacts with 10.
# # the infected people are in the center of grid (three options: 'center', 'corner', 'rand')
'''
pt = []
ratio = []
for idx in range(30):
    k = 0.075
    b = 1
    N = 10000
    t = 100
    s0 = 0.9999
    p = idx * 0.02 / 30
    q = (b / N / pi) ** 0.5
    s, i, r, _ = sir_spatial_model.simulation(b, k, t, N, s0, p, q, infected_pos = 'center', ifscatter = False)
    tol = [i + j for i, j in zip(i, r)]
    pt.append(p)
    ratio.append(max(tol))
    #times = range(t + 1)
plt.figure()
plt.plot(pt, ratio, c="blue")
    #plt.plot(times, tol, c="blue")
    #plt.plot(times, r, c="green")
plt.title("k = 0.075, b = 1, N = 10000, s0 = 0.9999")
plt.xlabel("p")
plt.ylabel("Fraction of people once infected")
#plt.legend(("Susceptible", "Infected", "Recovered"))
#plt.savefig('Discrete_spatial.png')
'''
fig, axs = plt.subplots(1, 3)
fig.suptitle('Plots of susceptible people for 3 different initializations (p = 0.01, s0 = 0.999)')
k = 0.075
b = 1
N = 10000
t = 100
s0 = 0.999
p = 0.01
q = (b / N / pi) ** 0.5
s, i, r, populations = sir_spatial_model.simulation(b, k, t, N, s0, p, q, infected_pos = 'center', ifscatter = False)
x = []
y = []
for i in range(N):
    if populations[i].get_state() == 'S':
        x.append(populations[i].get_pos()[0, 0])
        y.append(populations[i].get_pos()[0, 1])
axs[0].scatter(x, y, marker='o', color = 'b')
axs[0].set_title('center')

s, i, r, populations = sir_spatial_model.simulation(b, k, t, N, s0, p, q, infected_pos = 'corner', ifscatter = False)
x = []
y = []
for i in range(N):
    if populations[i].get_state() == 'S':
        x.append(populations[i].get_pos()[0, 0])
        y.append(populations[i].get_pos()[0, 1])
axs[1].scatter(x, y, marker='o', color = 'b')
axs[1].set_title('corner')

s, i, r, populations = sir_spatial_model.simulation(b, k, t, N, s0, p, q, infected_pos = 'rand', ifscatter = False)
x = []
y = []
for i in range(N):
    if populations[i].get_state() == 'S':
        x.append(populations[i].get_pos()[0, 0])
        y.append(populations[i].get_pos()[0, 1])
axs[2].scatter(x, y, marker='o', color = 'b')
axs[2].set_title('random')

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()