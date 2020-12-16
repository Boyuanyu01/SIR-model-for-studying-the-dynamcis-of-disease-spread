import sys
sys.path.append("../sir/")
import sirv_discrete as sd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set

# 1.1:
# Simulation 1: N =100 = 100*100 grid; t = 100; i0 = 0.001; a = True (random activity); k =0.05; r = 0.5;
# There are 100 * 100 agents with the initial infection rate of 0.001. Each agent's activity level (from 0 to 8) is randomly assigned.
# Recovery rate among the infected is 0.05. The chance of getting infected upon contacting infected agent is 0.5.

n = 100; t = 100; i0 = 0.001; a = True; k =0.05; r = 0.5;
ds1 = sd.disease_spread(n,t,i0,a,k,r)

# 1.2:
# Simulation 2: N =100 = 100*100 grid; t = 100; i0 = 0.001; a = True (random activity); k =0.05; r = 0.5; vac = 0.005
# Same condition as Simulation 1 except that we now have vaccine

n = 100; t = 100; i0 = 0.001; a = True; k =0.05; r = 0.5
ds2 = sd.disease_spread(n,t,i0,a,k, vac=0.005)


# Plot and save 2 by 2 subplots showing results from above two simulations.
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(range(t+1), ds1[:,0], label = 'S', c ='b')
axs[0, 0].plot(range(t+1), ds1[:,1], label = 'I', c = 'r')
axs[0, 0].plot(range(t+1), ds1[:,2], label = 'R', c = 'g')
axs[0, 0].set_title("no vaccine, SIR")
axs[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)

axs[1, 0].plot(range(t+1), ds2[:,0], label = 'S', c ='b')
axs[1, 0].plot(range(t+1), ds2[:,1], label = 'I', c = 'r')
axs[1, 0].plot(range(t+1), ds2[:,2], label = 'R', c = 'g')
axs[1, 0].plot(range(t+1), ds2[:,3], label = 'V', c = 'c')
axs[1, 0].set_title("w/ vaccine, SIRV")
#axs[1,0].legend()
axs[1,0].legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)

axs[0, 1].plot(range(t+1), ds1[:,0]+ds1[:,2]+ds1[:,3], label = 'Healthy', c ='g')
axs[0, 1].plot(range(t+1), ds1[:,1], label = 'I', c = 'r')
axs[0, 1].set_title("no vaccine, H vs. I")

axs[1, 1].plot(range(t+1), ds2[:,0]+ds2[:,2]+ds2[:,3], label = 'Healthy', c ='g')
axs[1, 1].plot(range(t+1), ds2[:,1], label = 'Infected', c = 'r')
axs[1, 1].set_title("w/ vaccine, H vs. I")
axs[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
fig.tight_layout()

plt.legend()
fig.tight_layout()
plt.savefig('Disease_spread_time.png')

plt.show()


# 2: Get a list of activity levels of agents with a certain state. Do this for all four states.

n = 100; k =0.05; i0 = 0.001; t = 100
pop = sd.population(100,0.001,True)
# get
S = np.zeros((101,9)); I = np.zeros((101,9)); R = np.zeros((101,9)); V = R = np.zeros((101,9))
for i in range(t+1):
    actS = sd.inf_vs_act(pop,state='S')
    actI = sd.inf_vs_act(pop,state='I')
    actR = sd.inf_vs_act(pop,state='R')
    actV = sd.inf_vs_act(pop,state='V')
    S[i] = actS; I[i] = actI; R[i] = actR; V[i] = actV
    pop = sd.simulate(pop,k=0.05, r=0.5)


# Plot above results for states 'S' and 'I' (more interesting than 'R' or 'V')

f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
g1 = sns.heatmap(S,cmap="rocket",cbar=True,ax=ax1)
g1.set_title('Susceptible')
g1.set_ylabel('time')
g1.set_xlabel('activity level')
g2 = sns.heatmap(I,cmap="rocket",cbar=True,ax=ax2)
g2.set_title('Infected')
g2.set_ylabel('time')
g2.set_xlabel('activity level')

f.tight_layout()
plt.savefig('Activity_vs_states.png')


# 3.1:
# Grid averaging. Average state grids over a specified time interval (e.g. every t=10)
# This is no-vaccine condition.

n = 100; k =0.05; i0 = 0.001; t = 100
pop = sd.population(100,0.001,True)

state_grids = []
avg_grids = []
for i in range(t+1):
    pop = sd.simulate(pop,k=0.05, r=0.5)
    state_grids.append(sd.state_grid(pop))
for n in range(0,10):
    tbin = [1,10]
    ntbin = [n*10+t for t in tbin]
    avg_grid = sum(state_grids[ntbin[0]:ntbin[1]])/10
    avg_grids.append(avg_grid)

# Plot results from above. Only show 4 equi-spaced grid-averages (there are 10 averaged grids but only show ones with index 0,3,6,9).

f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharey=True)

g1 = sns.heatmap(avg_grids[0],cmap="rocket",cbar=True,ax=ax1)
g1.set(xticklabels=[]); g1.axes.get_yaxis().set_visible(False)
g1.set_title("Avg. of t1-t10")
g2 = sns.heatmap(avg_grids[3],cmap="rocket",cbar=True,ax=ax2)
g2.set(xticklabels=[]); g2.axes.get_yaxis().set_visible(False)
g2.set_title("Avg. of t31-t40")
g3 = sns.heatmap(avg_grids[6],cmap="rocket",cbar=True,ax=ax3)
g3.set(xticklabels=[]); g3.axes.get_yaxis().set_visible(False)
g3.set_title("Avg. of t61-t70")
g4 = sns.heatmap(avg_grids[9],cmap="rocket",cbar=True,ax=ax4)
g4.set(xticklabels=[]); g4.axes.get_yaxis().set_visible(False)
g4.set_title("Avg. of t91-t100")

f.tight_layout()
plt.savefig('Disease_spread_novaccine.png')
plt.show()

# 3.2:
# Grid averaging. Average state grid over a specified time interval (e.g. every t=10)
# Same condition as before but with vaccine.

n = 100; k =0.05; i0 = 0.001; t = 100
pop = sd.population(100,0.001,True)

state_grids = []
avg_grids = []
for i in range(t+1):
    pop = sd.simulate(pop,k=0.05, r=0.5, v=0.005)
    state_grids.append(sd.state_grid(pop))
for n in range(0,10):
    tbin = [1,10]
    ntbin = [n*10+t for t in tbin]
    avg_grid = sum(state_grids[ntbin[0]:ntbin[1]])/10
    avg_grids.append(avg_grid)

# Plot results from above. Only show 4 equi-spaced grid-averages (there are 10 averaged grids but only show ones with index 0,3,6,9).

f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharey=True)

g1 = sns.heatmap(avg_grids[0],cmap="rocket",cbar=True,ax=ax1)
g1.set(xticklabels=[]); g1.axes.get_yaxis().set_visible(False)
g2 = sns.heatmap(avg_grids[3],cmap="rocket",cbar=True,ax=ax2)
g2.set(xticklabels=[]); g2.axes.get_yaxis().set_visible(False)
g3 = sns.heatmap(avg_grids[6],cmap="rocket",cbar=True,ax=ax3)
g3.set(xticklabels=[]); g3.axes.get_yaxis().set_visible(False)
g4 = sns.heatmap(avg_grids[9],cmap="rocket",cbar=True,ax=ax4)
g4.set(xticklabels=[]); g4.axes.get_yaxis().set_visible(False)

f.tight_layout()
plt.savefig('Disease_spread_withvaccine.png')
plt.show()
