import numpy as np
from math import floor
from math import ceil
import random
import matplotlib.pyplot as plt

class SIR_DISCRETE_VARIATION:
    """
    A class for an agent(person) in discrete SIR model
    """

    def __init__(self, state = 'S', pos = [0, 0]):
        """
        Initialize an agent with its inital state.
        """
        self.state = state
        self.pos = pos
        self.neighbors = []

    def __str__(self):
        return "SIR(state = {}, pos = [{}, {}])".format(self.state, self.pos[0], self.pos[1])

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
    def get_neighbors(self):
        """
        Get an agent's neighbors.
        """
        return self.neighbors
    def set_neighbors(self, neighbors):
        """
        Set an agent's neighbors.
        """
        self.neighbors = neighbors

def grid_form(L = 1, N = 1):
    '''
    Construct a grid and return row, column number and grid spacing
    '''
    row = int(floor(N ** 0.5))
    col = int(ceil(N / row))
    dlr = L / (row - 1)
    dlc = L / (col - 1)
    return row, col, dlr, dlc

def grid_position(row = 1, col = 1, dlr = 1, dlc = 1, idx = 0):
    '''
    Return the position of the agent on the grid
    '''
    i = idx // row
    j = idx - i * row
    return [i * dlc, j * dlr]

def cal_distance(pos1 = [0, 0], pos2 = [1, 1]):
    '''
    Calculate the distance between two agent used to check if they are withing the square of 2D0
    '''
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

def population(N, s0, i0):
    """
    Initializa a population of N people with a given ratio of 'S', 'I' and 'R' states and positions on the 2D grid

    Parameters:
    - N: Number of people in the population
    - i0: initial infection rate
    - s0: initial susceptibility rate

    Returns:
    - pop: a list of classes, in which each class represents a person with a state (S/I/R) and a position

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
        person = SIR_DISCRETE_VARIATION(state)
        pops.append(person)
    
    L = 1.0
    row, col, dlr, dlc = grid_form(L = L, N = N)
    count = 0
    # Randomly assign the agents to the grid
    for i in np.random.permutation(N):
        pops[count].set_pos(grid_position(row, col, dlr, dlc, i))
        count += 1

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

def find_neighbors(pops, D, idx):
    neighbors = []
    for i in range(len(pops)):
        if i != idx and cal_distance(pops[i].pos, pops[idx].pos) <= D:
            neighbors.append(i)
    return neighbors

def infected_range(D0, T0, T1, t):
    '''
    Calculate the length of square D within which people can infect others
    '''
    if t < T0:
        return D0
    elif t < T1:
        lam = 1.0
        return D0 * np.exp(- (t - T0) / lam)
    else:
        return D0

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

def update(tt, T0, D0, T1, pops, b, k):
    """
    Update the states of people in the population for one day of simulation

    Parameters:
    - pops: a list of classes, in which each class represents a person with a state (S/I/R)
    - b: number of contacts a day per person
    - k: recovery rate
    - tt: date
    - T0: the data 'lock down' begins
    - T1: the data 'lock down' ends
    - D0: initial length of square
    """
    infected = find_infection(pops)
    # Every infected people contact b people every day and infect susceptible people
    for i in infected:
        if tt > T0 and tt <= T1:
            # Update square length
            D = infected_range(D0, T0, T1, tt)
            # Find the agent's neighbors
            pops[i].set_neighbors(find_neighbors(pops, D, i))
        length = len(pops[i].get_neighbors())
        # Up to b people can be infected
        if length <= b:
            candidates = pops[i].get_neighbors()
        else:
            candidates = random.sample(pops[i].get_neighbors(), b)
        for j in candidates:
            if(pops[j].get_state() == 'S'):
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
        x.append(pops[i].get_pos()[0])
        y.append(pops[i].get_pos()[1])
    plt.figure()
    plt.scatter(x, y, marker='o', color = 'r')
    plt.title('time at t = {}'.format(t))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('../doc/final/infected_grid{}.png'.format(t))

def simulation(b, k, t, N, s0, i0, T0, D0, T1, ifscatter = False, time_points = [1, 2, 3, 5, 20]):
    """
    Simulate the discrete SIR model for t days

    Parameters:

    - b: number of contacts a day per person
    - k: recovery rate
    - t: number of days to run simulation
    - N: Number of people in the population
    - i0: initial infection rate
    - s0: initial susceptibility rate
    - tt: date
    - T0: the data 'lock down' begins
    - T1: the data 'lock down' ends
    - D0: initial length of square
    - ifscatter: whether make scatter plot or not
    - time_points: the dates for scatter plot to be plotted
    
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
    
    T0 = T0
    D0 = D0
    T1 = T1
    
    if ifscatter:
        make_scatterplot(pops, 0)
    
    for idx in range(N):
        pops[idx].set_neighbors(find_neighbors(pops, D0, idx))
        
    # update the population for t days
    for tt in range(1, t + 1):
        pops, st, it, rt = update(tt, T0, D0, T1, pops, b, k)
        if ifscatter and tt in time_points:
            make_scatterplot(pops, tt)
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
