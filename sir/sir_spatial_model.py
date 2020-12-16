import numpy as np
from numpy.linalg import norm
from math import floor
from math import pi
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

class SIR_SPATIAL_MODEL:
    """
    A class for an agent(person) in discrete SIR model with spacial structure
    """

    def __init__(self, state = 'S', pos = np.random.uniform(0, 1, (1, 2))):
        """
        Initialize an agent with its inital state.
        """
        self.state = state
        self.pos = pos

    def __str__(self):
        return "SIR(state = {}, pos = [{}, {}])".format(self.state, self.pos[0, 0], self.pos[0, 1])

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

    def get_pos(self):
        """
        Get an agent's position.
        """
        return self.pos
    
    def set_pos(self, pos):
        """
        Change an agent's position.
        """
        self.pos = pos

def population(N, s0, infected_pos = 'rand'):
    """
    Initializa a population of N people with a given ratio of 'S', 'I' and 'R' states and positions on the 2D grid

    Parameters:
    - N: Number of people in the population
    - s0: initial susceptibility rate
    - infected_pos: where to put the infected agents
    
    Returns:
    - pop: a list of classes, in which each class represents a person with a state (S/I/R) and a position

    """
    pops = []
    snum = floor(N * s0)
    infectednum = N - snum
    for i in range(N):
        pos = np.random.uniform(0, 1, (1, 2))
        if i < infectednum:
            state = 'I'
            if infected_pos == 'corner':
                pos = np.random.uniform(0, 1 / 32, (1, 2))
            elif infected_pos == 'center':
                pos = np.random.uniform(31 / 64, 33 / 64, (1, 2))
            else:
                pos = np.random.uniform(0, 1, (1, 2))
        else:
            state = 'S'
        person = SIR_SPATIAL_MODEL(state, pos)
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

def moving_around(pops, p):
    '''
    Moving every agent by a random vetor p
    '''
    N = len(pops)
    for i in range(N):
        dpos = np.random.randn(1, 2)
        dpos = dpos / norm(dpos)
        dpos = dpos * p
        temp = pops[i].get_pos() + dpos
        if(temp[0, 0] < 0 or temp[0, 0] > 1 or temp[0, 1] < 0 or temp[0, 1] > 1):
            continue
        else:
            pops[i].set_pos(temp)
    
    return pops


def update(pops, b, k, p, q):
    """
    Update the states of people in the population for one day of simulation

    Parameters:
    - pops: a list of classes, in which each class represents a person with a state (S/I/R)
    - b: number of contacts a day per person
    - k: recovery rate
    - p: moving vector
    """

    N = len(pops)

    # Agents move around
    pops = moving_around(pops, p)
    
    # Combine position data
    X = pops[0].get_pos()
    for i in range(1, N):
        X = np.vstack((X, pops[i].get_pos()))

    # Construct the KDtree 
    tree = KDTree(X)

    # Find infected agents
    infected = find_infection(pops)

    for i in infected:
        # Find neighbors within radius q
        candidates = tree.query_radius(X[i].reshape(1, -1), r = q)   
        for j in candidates[0]:
            if(j != i and pops[j].get_state() == 'S'):
                #only susceptible individuals are infected
                pops[j].set_state('I')


    # k infected people become removed every day
    infected = find_infection(pops)
    removed_number = floor(k * len(infected))
    for i in range(removed_number):
        pops[infected[i]].set_state('R')

    susceptible_ratio, infected_ratio, removed_ratio = statistics(pops)
    return pops, susceptible_ratio, infected_ratio, removed_ratio

def make_scatterplot(pops, t):
    '''
    Visualize the infected people on the grid at day t by scatter plot
    '''
    infected = find_infection(pops)
    x = []
    y = []
    for i in infected:
        x.append(pops[i].get_pos()[0, 0])
        y.append(pops[i].get_pos()[0, 1])
    plt.figure()
    plt.scatter(x, y, marker='o', color = 'r')
    plt.title('time at t = {}'.format(t))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #plt.savefig('check_infected_agents_spatial_distribution{}.png'.format(t))

def simulation(b, k, t, N, s0, p, q, infected_pos = 'rand', ifscatter = False, time_points = [1, 2, 3, 5, 20]):
    """
    Simulate the discrete SIR model for t days

    Parameters:

    - b: number of contacts a day per person
    - k: recovery rate
    - t: number of days to run simulation
    - N: Number of people in the population
    - s0: initial susceptibility rate
    - p : moving vector
    - ifscatter: whether make scatter plot or not
    - time_points: the dates for scatter plot to be plotted
    
    Returns:

    - s: time series, ratios of the susceptible to population
    - i: time series, ratios of the infected to population
    - r: time series, ratio of the removed to population
    """
    # initial the population
    pops = population(N, s0, infected_pos)
    # initial the statistics
    st0, it0, rt0 = statistics(pops)
    s = [st0]
    i = [it0]
    r = [rt0]
    
    p = p
    
    if ifscatter:
        make_scatterplot(pops, 0)
        
    # update the population for t days
    for tt in range(1, t + 1):
        pops, st, it, rt = update(pops, b, k, p, q)
        if ifscatter and tt in time_points:
            make_scatterplot(pops, tt)
        s.append(st)
        i.append(it)
        r.append(rt)
    return s, i, r, pops
