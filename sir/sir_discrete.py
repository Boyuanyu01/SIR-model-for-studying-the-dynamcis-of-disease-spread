import numpy as np
from math import floor

class SIR_DISCRETE:
    """
    A class for an agent(person) in discrete SIR model
    """

    def __init__(self, state = 'S'):
        """
        Initialize an agent with its inital state.
        """
        self.state = state

    def __str__(self):
        return "SIR(state = {})".format(self.state)

    def __repr__(self):
        return self.__str__()

    def get_state(self):
        """
        Get an agent's state.
        """
        return self.state

    def set_state(self, state):
        """
        Change an agent's state.
        """
        self.state = state

def population(N, s0, i0):
    """
    Initializa a population of N people with a given ratio of 'S', 'I' and 'R' states

    Parameters:
    - N: Number of people in the population
    - i0: initial infection rate
    - s0: initial susceptibility rate

    Returns:
    - pop: a list of classes, in which each class represents a person with a state (S/I/R)

    """
    pops = []
    infectednum = floor(N * i0)
    susceptiblenum = floor(N * s0)
    for i in range(N):
        if i < infectednum:
            state = 'I'
        elif i < infectednum + susceptiblenum:
            state = 'S'
        else:
            state = 'R'
        person = SIR_DISCRETE(state)
        pops.append(person)

    return pops

def find_infection(pops):
    """
    Find the infected people

    Parameters:
    - pop: a list of classes, in which each class represents a person with a state (S/I/R)

    Returns:
    - infected: a list of indices of the infected people

    """
    infected = []
    N = len(pops)
    for i in range(N):
        if pops[i].get_state() == 'I':
            infected.append(i)
    return infected

def statistics(pops):
    """
    Calculate the ratio of susceptible, infected, and removed people in the population

    Parameters:
    - pops: a list of classes, in which each class represents a person with a state (S/I/R)

    Returns:
    - susceptible_ratio: ratio of the susceptible to population
    - infected_ratio: ratio of the infected to population
    - removed_ratio: ratio of the removed to population

    """
    susceptible = 0
    infected = 0
    removed = 0
    for person in pops:
        if person.get_state() == 'S':
            susceptible = susceptible + 1
        if person.get_state() == 'I':
            infected = infected + 1
        if person.get_state() == 'R':
            removed = removed + 1
    susceptible_ratio = susceptible / len(pops)
    infected_ratio = infected / len(pops)
    removed_ratio = removed / len(pops)
    return susceptible_ratio, infected_ratio, removed_ratio

def update(pops, b, k):
    """
    Update the states of people in the population for one day of simulation

    Parameters:
    - pops: a list of classes, in which each class represents a person with a state (S/I/R)
    - b: number of contacts a day per person
    - k: recovery rate

    """
    infected = find_infection(pops)
    N = len(pops)
    # Every infected people contact b people every day and infect susceptible people
    for i in infected:
        for j in range(floor(b)):
            idx = floor(N * np.random.rand())
            while(idx == i):
                #reset a index as people cannot be in contact with themselves
                idx = floor(N * np.random.rand())
            if(pops[idx].get_state() == 'S'):
                #only susceptible individuals are infected
                pops[idx].set_state('I')

    # k infected people become removed every day
    infected = find_infection(pops)
    removed_number = floor(k * len(infected))
    for i in range(removed_number):
        pops[infected[i]].set_state('R')

    susceptible_ratio, infected_ratio, removed_ratio = statistics(pops)
    return pops, susceptible_ratio, infected_ratio, removed_ratio

def simulation(b, k, t, N, s0, i0):
    """
    Simulate the discrete SIR model for t days

    Parameters:

    - b: number of contacts a day per person
    - k: recovery rate
    - t: number of days to run simulation
    - N: Number of people in the population
    - i0: initial infection rate
    - s0: initial susceptibility rate

    Returns:

    - s: time series, ratios of the susceptible to population
    - i: time series, ratios of the infected to population
    - r: time series, ratio of the removed to population
    """
    # initial the population
    pops = population(N, s0, i0)
    # initial the statistics
    st0, it0, rt0 = statistics(pops)
    s = [st0]
    i = [it0]
    r = [rt0]
    # update the population for t days
    for tt in range(t):
        pops, st, it, rt = update(pops, b, k)
        s.append(st)
        i.append(it)
        r.append(rt)
    return s, i, r

def checksifs(s, target):
    """
    Check if s(t) = target for some t and return t if it is True
    """
    checks = False
    t0 = len(s)
    count = 0
    for si in s:
        if si - target < 1e-4:
            checks = True
            t0 = count
            break
        count = count + 1
    return checks, t0

def checksifi(i, target):
    """
    Check if i(t) > target for some t and return t if it is True
    """
    checki = False
    t0 = len(i)
    count = 0
    for ii in i:
        if ii > target:
            checki = True
            t0 = count
            break
        count = count + 1
    return checki, t0
