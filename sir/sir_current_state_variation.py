import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SIR_ODE:
    """
    A class for a SIR initial value problem
    """

    def __init__(self, k, b, s, i, r, N=1):
        """
        Initialize a SIR_ODE object with parameters governing infection dynamics as well as initial data.
        Since the simulation would might start from the condition that is left off by a previous simulation
        all of S, I and R has to be accepted as inputs
        """

        self.N = N #size of population, default running with normalized population
        self.k = k #"rate of recovery"
        self.b = b #"rate of infection"
        self.sir0 = np.array([s, i, r]) #initial condition array of Susceptible, Infectious, Removed in proportion
        self.SIR0 = N*self.sir0 #the population size of initial condition

    def __str__(self):
        return "SIR(k={0}, b={1}, i0={2}, N={3})".format(self.k, self.b, self.sir0[1], self.N)
    
    def __repr__(self):
        return self.__str__()
    
    def derivative(self, data, t=0):
        """
        Returns the derivative at given time and data [s,i,r] for normalized version only.
        """
        data = np.array(data)
        s = data[0]
        i = data[1]
        fprime = np.array([-1*self.b*s*i, self.b*s*i-self.k*i, self.k*i])

        return fprime
    
    def set_ivp(self, s, i, r, N):
        """
        Sets the initial values of this ODE system
        """
        self.N = N
        self.sir0 = np.array([s, i, r])
        self.SIR0 = N*self.sir0
    
    def set_param(self, k, b):
        """
        Sets the parameters k and b of this ODE system
        """
        self.k = k
        self.b = b

    def simulate(self, teval, figure=False, normalized_plot=True, **kwargs):
        """
        Simulates this ODE for given times in teval. Returns the solution as well as a plot of the simulation if figure=True
        """

        # Simulate
        events = kwargs.get("events", None)
        f = lambda t, sir : np.array([-1*self.b*sir[0]*sir[1], self.b*sir[0]*sir[1]-self.k*sir[1], self.k*sir[1]])
        solution = solve_ivp(f, t_span=(teval[0], teval[-1]), y0=self.sir0, t_eval=teval, events=events)

        if figure:
            # Plot the resulting solution
            plt.figure(figsize=(8, 8))
            plt.plot(solution.t, (self.N*(1-normalized_plot) + normalized_plot)*solution.y[0], c="blue")
            plt.plot(solution.t, (self.N*(1-normalized_plot) + normalized_plot)*solution.y[1], c="red")
            plt.plot(solution.t, (self.N*(1-normalized_plot) + normalized_plot)*solution.y[2], c="green")
            plt.title("SIR starts k = {0}, b = {1}, N = {2}, (S,I,R) = {3}".format(self.k,self.b,self.N,self.sir0),fontsize=18)
            plt.xlabel("Time", fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel("Population", fontsize=24)
            plt.legend(("Susceptible", "Infected", "Recovered"), fontsize=18)

            return (solution, plt)
        else:
            return (solution, None)

def SIR_continuous_state(s,i,r,b,k,func,t_max=100,n=1000000):
    """
    function that simulate a continuous change in b
    as a function of i
    """
    s_list = [s] #forming list for storing past values
    i_list = [i]
    r_list = [r]
    step = t_max/n #step size for simulation
    for t in range(n-1):
        new_b = func(i_list[-1]) #b changes according to current state of i
        ds = -new_b*s_list[-1]*i_list[-1]
        di = new_b*s_list[-1]*i_list[-1] - k*i_list[-1]
        dr = k*i_list[-1] #derivative calculated
        new_s = s_list[-1] + ds * step #derive new state based on previous one
        new_i = i_list[-1] + di * step
        new_r = r_list[-1] + dr * step
        s_list.append(new_s) #append for storage
        i_list.append(new_i)
        r_list.append(new_r)
    return s_list,i_list,r_list