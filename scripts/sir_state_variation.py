import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append("../sir/")
import sir_state_variation
from sir_current_state_variation import SIR_continuous_state

#This script will run experiment and plot result for the propsed variation
#on SIR model that depends on current state
def large_number_infected(t,y):
    """
    event set to alarm the public when there is large number of population infected
    result change in b
    """
    return y[1]-0.3
def large_number_cured(t,y):
    """
    event set to signal that a large number of population is recovered
    medical supplies are allocated
    """
    return y[2]-0.4
large_number_infected.terminal=True #both events terminate the simulation
large_number_cured.terminal=True
k = 0.05 #initial condition
b = 0.8
N = 1
i = 0.000001
s = 1-i
r=0
ode1 = sir_state_variation.SIR_ODE(k, b, s, i, r, N)
teval = np.linspace(0, 80, 10000)
(sol1, obj1) = ode1.simulate(teval, figure=False, normalized_plot=True, events=large_number_infected)

#second simulation would start from the condition left off by the first simulation
s2 = sol1.y[0][-1] 
i2 = sol1.y[1][-1]
r2 = sol1.y[2][-1]
b2 = b / 8 #change in b as a result of the event
time_elapsed = len(sol1.t) #continue second simulation based on time step that is left off
teval2 = np.linspace(sol1.t[-1], 80, 10000-time_elapsed)
ode2 = sir_state_variation.SIR_ODE(k, b2, s2, i2, r2, N)
(sol2, obj2) = ode2.simulate(teval2, figure=False, normalized_plot=True, events=large_number_cured)

#third simulation would start from the condition left off by the second simulation
s3 = sol2.y[0][-1] 
i3 = sol2.y[1][-1]
r3 = sol2.y[2][-1]
k2 = k * 2.5 #change in k as a result of the event
time_elapsed2 = len(sol2.t) #continue second simulation based on time step that is left off
teval3 = np.linspace(sol2.t[-1], 80, 10000-time_elapsed-time_elapsed2)
ode3 = sir_state_variation.SIR_ODE(k2, b2, s3, i3, r3, N)
(sol3, obj3) = ode3.simulate(teval3, figure=False, normalized_plot=True)

sol_s = np.concatenate((sol1.y[0],sol2.y[0],sol3.y[0]))
sol_i = np.concatenate((sol1.y[1],sol2.y[1],sol3.y[1]))
sol_r = np.concatenate((sol1.y[2],sol2.y[2],sol3.y[2]))# concatenate the solutions
fig, ax = plt.subplots(figsize=(10,3)) 
plt.plot(teval, sol_s, c="blue", label="Susceptible")
plt.plot(teval, sol_i, c="red", label="Infected")
plt.plot(teval, sol_r, c="green", label="Recovered") #plot concatenated versions of the simulation
plt.title("SIR model with sharp changes in b and k in the process",fontsize=18)
plt.xlabel("Time", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Population", fontsize=15)
plt.legend(fontsize=15)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("../doc/final/phase_transition_full_infection.png")



"""
the following plot would examine, fixing the initial value of b and for various maginitude in change of b at the
second stage, the relationship between the time for such alternation to take place and the maximum infected population
in the entire simluation.
"""
k = 0.05 #initial condition
b = 0.8
N = 1
i = 0.000001
s = 1-i
r=0
teval = np.linspace(0, 80, 10000)
time_trigger = np.linspace(10, 25, 300)
fig, ax = plt.subplots(figsize=(10,3))
for change_b in range(9):
    max_infected = []
    for time in time_trigger:
        def large_number_infected(t,y):
            """
            define a event set to alarm the public when some time has passed and 
            significant number of people are infected
            result change in b
            """
            return t-time
        large_number_infected.terminal=True
        ode1 = sir_state_variation.SIR_ODE(k, b, s, i, r, N)
        (sol1, obj1) = ode1.simulate(teval, figure=False, normalized_plot=True, events=large_number_infected)
        s2 = sol1.y[0][-1] 
        i2 = sol1.y[1][-1]
        r2 = sol1.y[2][-1]
        b2 = b / (change_b+2)
        time_elapsed = len(sol1.t) #continue second simulation based on time step that is left off
        teval2 = np.linspace(sol1.t[-1], 80, 10000-time_elapsed)
        ode2 = sir_state_variation.SIR_ODE(k, b2, s2, i2, r2, N)
        (sol2, obj2) = ode2.simulate(teval2, figure=False, normalized_plot=True)
        sol_i = np.concatenate((sol1.y[1],sol2.y[1]))
        max_i = max(sol_i)
        max_infected.append(max_i)
    plt.plot(time_trigger, max_infected, label=f"b2 = b/{change_b+2}")
plt.title("Relationship between Time Triggers Change in b and Maximum Infection Rate",fontsize=16)
plt.xlabel("Time", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Maximum I", fontsize=15)
plt.legend(fontsize=15)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("../doc/final/b_and_max_i.png")

"""
Now we want more freedom of b in response to change in the size of infected population. 
We set the values of b as a function of i at every step
"""
k = 0.05 #initial condition
b = 0.8
N = 1
i = 0.000001
s = 1-i
r=0
teval = np.linspace(0,80,1000000)
def linear_b(i):
    """
    b as a linear functin of current state i
    """
    return max(b - 2*i,0.01) #avoid having b less than 0
def quadratic_b(i):
    """
    b as a quadratic functin of current state i
    """
    return max(b - 5*i**2,0.01)
def exp_b(i):
    """
    b as a exponential functin of current state i
    """
    return max(b - 2**i +1,0.01)
def reciprocal_b(i):
    """
    b as a reciprocal functin of polynomial of i
    """
    return max(b/(1+10*i),0.01)

fig, ax = plt.subplots(1,4,figsize=(15,3),sharey=True)
s_list,i_list,r_list = SIR_continuous_state(s,i,r,b=0.8,k=0.05,t_max=80,n=1000000, func=linear_b)
ax[0].plot(teval, s_list, c="blue", label="Susceptible")
ax[0].plot(teval, i_list, c="red", label="Infected")
ax[0].plot(teval, r_list, c="green", label="Recovered")
ax[0].set_title("b = 0.8 - 2i")
s_list,i_list,r_list = SIR_continuous_state(s,i,r,b=0.8,k=0.05,t_max=80,n=1000000, func=quadratic_b)
ax[1].plot(teval, s_list, c="blue", label="Susceptible")
ax[1].plot(teval, i_list, c="red", label="Infected")
ax[1].plot(teval, r_list, c="green", label="Recovered")
ax[1].set_title("b = 0.8 - 5i^2")
s_list,i_list,r_list = SIR_continuous_state(s,i,r,b=0.8,k=0.05,t_max=80,n=1000000, func=exp_b)
ax[2].plot(teval, s_list, c="blue", label="Susceptible")
ax[2].plot(teval, i_list, c="red", label="Infected")
ax[2].plot(teval, r_list, c="green", label="Recovered")
ax[2].set_title("b = 0.8 - 2^i +1")
s_list,i_list,r_list = SIR_continuous_state(s,i,r,b=0.8,k=0.05,t_max=80,n=1000000, func=reciprocal_b)
ax[3].plot(teval, s_list, c="blue", label="Susceptible")
ax[3].plot(teval, i_list, c="red", label="Infected")
ax[3].plot(teval, r_list, c="green", label="Recovered")
ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax[3].set_title("b = 0.8/(1+10i)")
plt.savefig("../doc/final/phase_transition_full_infection.png")


