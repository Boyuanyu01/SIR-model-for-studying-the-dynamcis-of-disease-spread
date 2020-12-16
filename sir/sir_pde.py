import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.sparse as sparse
from matplotlib.animation import ArtistAnimation


class SIR_PDE:
    """
    A class for a SIR PDE model
    """

    def __init__(self, k, b, p, i0, M=200):
        """
        Initialize a SIR_PDE object with parameters governing infection dynamics as well as initial data.
        """

        self.M = M #size of grid (M x M) 
        self.k = k #"rate of recovery"
        self.b = b #"rate of infection"
        self.p = p #"diffusion weight"
        
        #initial condition array of Susceptible, Infectious, Removed in proportion
        self.s0 = np.ones((M,M)) - i0
        self.i0 = i0
        self.r0 = np.zeros((M,M))

        self.sir0 = self.s0.flatten()
        self.sir0 = np.append(self.sir0, self.i0.flatten())
        self.sir0 = np.append(self.sir0, self.r0.flatten())

        self.Sind = np.arange(start=0, stop=M**2)
        self.Iind = np.arange(start=M**2,stop=(2*(M**2)))
        self.Rind = np.arange(start=(2*(M**2)),stop=(3*(M**2)))

        # Construct Laplacian operator
        # First, construct first order difference matrix: Code from Brad Nelson's CAAM 37830
        def forward_diff_matrix(n):
            data = []
            i = []
            j = []
            for k in range(n - 1):
                i.append(k)
                j.append(k)
                data.append(-1)

                i.append(k)
                j.append(k+1)
                data.append(1)
                
            # we'll just set the last entry to 0 to have a square matrix
            return sparse.coo_matrix((data, (i,j)), shape=(n, n)).tocsr()
            
        D = forward_diff_matrix(M) / M
        D2 = -D.T @ D
        D2x = sparse.kron(sparse.eye(M), D2).tocsr()
        D2y = sparse.kron(D2, sparse.eye(M)).tocsr()
        self.L = D2x + D2y


    def __str__(self):
        return "SIR_PDE(k={0}, b={1}, p={2})".format(self.k, self.b, self.p)
    
    def __repr__(self):
        return self.__str__()
    
    def time_derivative(self, data, t=0):
        """
        Returns the derivative at given time and data [s,i,r] for normalized version only.
        """
        data = np.array(data)
        s = data[0]
        i = data[1]
        fprime = np.array([-1*self.b*np.multiply(s, i) + self.p*self.L @ s, self.b*np.multiply(s,i)-self.k*i + self.p*self.L @ i, self.k*i + self.p*self.L @ s])

        return fprime
    
    def set_ivp(self, i0, M=200):
        """
        Sets the initial values of this PDE system
        """
        self.s0 = np.ones((M,M)) - i0
        self.i0 = i0
        self.r0 = np.zeros((M,M))
    
    def set_param(self, k, b, p):
        """
        Sets the parameters k and b of this ODE system
        """
        self.k = k
        self.b = b
        self.p = p

    def simulate(self, teval, figure=False, **kwargs):
        """
        Simulates this ODE for given times in teval. Returns the solution as well as a plot of the simulation if figure=True
        """

        # Simulate
        events = kwargs.get("events", None)
        f = lambda t, sir : np.array(np.append(-1*self.b*np.multiply(sir[self.Sind],sir[self.Iind]) + self.p*self.L @ sir[self.Sind], [self.b*np.multiply(sir[self.Sind],sir[self.Iind])-self.k*sir[self.Iind] + self.p*self.L @ sir[self.Iind], self.k*sir[self.Iind] + self.p*self.L @ sir[self.Sind]]))
        solution = solve_ivp(f, t_span=(teval[0], teval[-1]), y0=self.sir0, t_eval=teval, events=events)

        if figure:
            # Plot the infected trajectory
            ifig=plt.figure(figsize=(16,10))
            ifig.set_tight_layout(True)
            ims = []
            for time in np.arange(start=0, stop=teval.shape[0], dtype=int):
                im = plt.imshow(solution.y[self.Iind, time].reshape((self.M, self.M)), animated=True)
                ims.append([im])
                
            plt.axis("off")
            plt.title("Infected - SIR PDE with k = {0}, b = {1}, p = {2}".format(self.k, self.b, self.p), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            animi = ArtistAnimation(ifig, ims)

            # Plot the susceptible trajectory
            sfig = plt.figure(figsize=(16,10))
            sfig.set_tight_layout(True)
            sims = []
            for time in np.arange(start=0, stop=teval.shape[0], dtype=int):
                sim = plt.imshow(solution.y[self.Sind, time].reshape((self.M, self.M)), animated=True)
                sims.append([sim])
            
            plt.axis("off")
            plt.title("Susceptible - SIR PDE with k = {0}, b = {1}, p = {2}".format(self.k, self.b, self.p), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            anims = ArtistAnimation(sfig, sims)

            # Plot the recovered trajectory
            rfig = plt.figure(figsize=(16,10))
            rfig.set_tight_layout(True)
            rims = []
            for time in np.arange(start=0, stop=teval.shape[0], dtype=int):
                rim = plt.imshow(solution.y[self.Rind, time].reshape((self.M, self.M)), animated=True)
                rims.append([rim])
            
            plt.axis("off")
            plt.title("Recovered - SIR PDE with k = {0}, b = {1}, p = {2}".format(self.k, self.b, self.p), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            animr = ArtistAnimation(rfig, rims)

            return (solution, animi, anims, animr)
        else:
            return (solution, None, None, None)