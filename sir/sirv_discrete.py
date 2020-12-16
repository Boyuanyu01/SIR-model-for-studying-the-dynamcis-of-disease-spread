import numpy as np
from math import floor
import itertools as it
from collections import Counter, OrderedDict
#from itertools import repeat
import sir_discrete as sd
import random

class SIRV_DISCRETE(sd.SIR_DISCRETE):
    def __init__(self, activity):
        super().__init__()
        self.activity = activity
        #self.met = met

    def get_activity(self):
        return self.activity

A = sd.SIR_DISCRETE()
B = SIRV_DISCRETE(activity = 8)


def population(n, i0, act = False):
    """
    Initializa a population of m*n people with a given ratio of 'S', 'I' and 'R' states and activity level

    Parameters:
    - N: Number of people in the population
    - i0: initial infection rate
    - act (Boolean): if True, agents are active and take a random activity level from range(9)
                     Default: False (set to 0).

    Returns:
    - pop: a list of classes, in which each class represents a person with a state (S/I/R/V)

    """
    pop = [[] for i in it.repeat(None, n)] # create nested list of m lists
    num_I = floor(n*n * i0)
    #num_S = floor(n*n * s0)
    for row in range(n):
        for col in range(n):
            if act == True:
                activity = np.random.choice(9)
                pop[row].append(SIRV_DISCRETE(activity))
            else:
                pop[row].append(SIRV_DISCRETE(0))
    ranidx = np.random.choice(a=n, size=(num_I,2), replace=True)
    #print(ranidx)
    for item in ranidx:
        pop[item[0]][item[1]].set_state('I')
    return pop


def statistics(pop):
    """
    Calculate the fraction of susceptible, infected, and removed people in the population

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)

    Returns:
    - sirv_frac: fraction of each state

    """
    sirv_states = [0,0,0,0]
    # flatten list of lists of sub-populations
    flat_pop = [item for subpop in pop for item in subpop]
    num_pop = len(flat_pop)
    for person in flat_pop:
        if person.get_state() == 'S':
            sirv_states[0] +=1
        elif person.get_state() == 'I':
            sirv_states[1] +=1
        elif person.get_state() == 'R':
            sirv_states[2] +=1
        elif person.get_state() == 'V':
            sirv_states[3] +=1
    sirv_frac = [state/num_pop for state in sirv_states]
    return sirv_frac


def contacts(pop,i,j):
    """
    Generate* possible contacts of a person positioned at [i,j] in the population grid.

    Parameters:
    - i: row index
    - j: column index
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)

    Yields:
    - indices of possible contacts of the person

    """
    flat_pop = [item for subpop in pop for item in subpop]
    n = int(np.sqrt(len(flat_pop)))
    inbrs = [-1, 0, 1]
    if i == 0:
        inbrs = [0, 1]
    if i == n-1:
        inbrs = [-1, 0]
    jnbrs = [-1, 0, 1]
    if j == 0:
        jnbrs = [0, 1]
    if j == n-1:
        jnbrs = [-1, 0]

    for delta_i in inbrs:
        for delta_j in jnbrs:
            if delta_i == delta_j == 0:
                continue
            yield i + delta_i, j + delta_j


def random_pos(pop):
    """
    Get shuffled list of indices of the population grid.

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)

    Returns:
    - ran_pos: a shuffled list of indices generated from all the combinations of n choose 2

    """
    flat_pop = [item for subpop in pop for item in subpop]
    n = int(np.sqrt(len(flat_pop)))
    ran_pos = list(it.product(range(n), repeat=2))
    random.shuffle(ran_pos)
    return ran_pos


def get_status(pop,i,j):
    """
    Get status of a person (SIRV state and activity level)

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)

    Returns:
    - state, activity

    """
    state = pop[i][j].get_state()
    activity = pop[i][j].get_activity()
    return state,activity


def update_status(pop,i,j,k,r):
    """
    Update the status (both activity and state) of contacts (SIRV state and activity level).
    Function assumes that the origin is infected,, and update all people contacted as infected.

    Parameters:
    - i: row index of the origin
    - j: column index of the origin
    - k: recovery rate
    - r: infection rate
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)

    Returns:
    - None

    """
    state_origin, act_origin = get_status(pop,i,j)
    for i2, j2 in contacts(pop,i,j):
        state_origin, act_origin = get_status(pop,i,j)
        state_cont, act_cont = get_status(pop,i2,j2)
        if state_cont == 'I' and np.random.rand() < k:
            pop[i2][j2].state = 'R'
        if act_origin == 0:
            break
        elif act_origin > 0 and act_cont > 0:
            pop[i2][j2].activity -=1
            if state_origin == 'I' and state_cont != 'V' and np.random.rand() > r:
                pop[i2][j2].state = 'I'
            pop[i][j].activity -=1
        elif act_origin > 0 and act_cont == 0:
            continue
    if state_origin == 'I' and np.random.rand() < k:
        pop[i][j].state = 'R'


def vaccinate(pop,v):
    """
    Change the states of random agents to 'V' if the agents' current states are either 'S' or 'R'

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)
    - v: fraction of population to be vaccinated

    Returns:
    -pop

    """
    flat_pop = [item for subpop in pop for item in subpop]
    N = len(flat_pop)
    num_vac = floor(N*v) # number of people to be vaccinated
    #print(num_vac)
    #i = 0
    for pos in random_pos(pop):
        #i+=1
        #print("inside the loop: ", i)
        state = pop[pos[0]][pos[1]].get_state()
        if num_vac == 0:
            #print("no vaccine")
            break
        elif state == 'S' or state == 'R':
            #print("This guy was: ", state)
            pop[pos[0]][pos[1]].state = 'V'
            #print("This guy is now: ", pop[pos[0]][pos[1]].get_state())
            num_vac -=1
            #print(num_vac)
        elif state == 'I' or state == 'V':
            continue
    return pop


def simulate(pop,k,r,v=0):
    """
    Run a single simulation (a single time step) of disease spread

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)
    - k: recovery rate
    - r: infection rate
    - v: fraction of population to be vaccinated

    Returns:
    -pop

    """

    activity = [] # list to store the original activity level of the population
    flat_pop = [item for subpop in pop for item in subpop]
    n = int(np.sqrt(len(flat_pop)))
    for person in flat_pop:
        activity.append(person.get_activity())
    # update people's states
    activity = np.array(activity).reshape((n,n))
    for pos in random_pos(pop):
        update_status(pop, pos[0],pos[1],k,r)
    # vaccinate people
    pop = vaccinate(pop,v)
    # recover the activity level of each person:
    for i in range(n):
        for j in range(n):
            pop[i][j].activity = activity[i][j]
    return pop


def inf_vs_act(pop,**kwargs):
    """
    Get a list of activity levels of agents whose states are given as a keyword argument

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)
    - kwargs: recovery rate

    Returns:
    -list_act: a list of activity levels

    """
    state = kwargs.get('state','I')
    n = len(pop)
    dict_act = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    idx = []
    act = []
    for i in range(n):
        for j in range(n):
            if pop[i][j].get_state() == state:
                idx.append((i,j))
    for agent in idx:
        act.append(pop[agent[0]][agent[1]].get_activity())
    total = len(idx)
    counted = Counter(act)
    d = {l: m/total for l, m in counted.items()}
    od = OrderedDict(sorted(d.items()))
    for a,f in od.items():
        dict_act.update({a:f})
    list_act = [item for key, item in dict_act.items()]
    return list_act


def state_grid(pop):
    """
    Get a grid whose elements are states of the population grid

    Parameters:
    - pop: a nested list of lists of classes, in which each class represents a person with a state (S/I/R/V)

    Returns:
    -state_grid

    """
    n = len(pop)
    state_grid = np.zeros((n,n))
    healthy = ['S','R','V']
    for i in range(n):
        for j in range(n):
            state = pop[i][j].get_state()
            if state == 'I':
                state_grid[i,j] = 1
            elif any(state for states in healthy):
                state_grid[i,j] = 0
    return state_grid


def disease_spread(n,t,i0,a=True,k=0.05,r=0.5,**kwargs):
    """
    Simulate the discrete SIRV model for t days

    Parameters:

    - b: number of contacts a day per person
    - k: recovery rate
    - t: number of days to run simulation
    - N: Number of people in the population
    - i0: initial infection rate
    - s0: initial susceptibility rate

    Returns:

    - s: time series, fractions of the susceptible to population
    - i: time series, fractions of the infected to population
    - r: time series, fractions of the removed to population
    """
    v = kwargs.get('vac', 0)

    # Initial population
    pop = population(n, i0, act = a)
    # array to store sirv_ratio
    sirv_fraction = np.zeros((t+1,4))
    sirv_fraction[0] = np.array(statistics(pop))
    # run the simulation t times
    for tbin in range(t):
        pop = simulate(pop, k=k, r=r, v=v)
        #print(statistics(pop))
        sirv_fraction[tbin+1] = np.array(statistics(pop))
    return sirv_fraction
