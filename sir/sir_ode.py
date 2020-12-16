import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SIR_ODE:
    """
    A class for a SIR initial value problem
    """

    def __init__(self, k, b, i0, N=1):
        """
        Initialize a SIR_ODE object with parameters governing infection dynamics as well as initial data.
        """

        self.N = N #size of population, default running with normalized population
        self.k = k #"rate of recovery"
        self.b = b #"rate of infection"
        self.sir0 = np.array([1-i0, i0, 0]) #initial condition array of Susceptible, Infectious, Removed in proportion
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
    
    def set_ivp(self, i0, N):
        """
        Sets the initial values of this ODE system
        """
        self.N = N
        self.sir0 = np.array([1-i0, i0, 0])
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
            plt.title("SIR with k = {0}, b = {1}, N = {2}, i0 = {3}".format(self.k, self.b, self.N, self.sir0[1]), fontsize=20)
            plt.xlabel("Time", fontsize=32)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylabel("Population", fontsize=32)
            plt.legend(("Susceptible", "Infected", "Recovered"), fontsize=20)

            return (solution, plt)
        else:
            return (solution, None)