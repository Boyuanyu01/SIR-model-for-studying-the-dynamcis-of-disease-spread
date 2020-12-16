import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.sparse as sparse
from matplotlib.animation import ArtistAnimation


class SIRS_PDE:
    """
    A class for a SIRS PDE model
    """

    def __init__(self, k, b, m, p, i0, M=200):
        """
        Initialize a SIRS_PDE object with parameters governing infection dynamics as well as initial data.
        """

        self.M = M #size of grid (M x M) 
        self.k = k #"rate of recovery"
        self.b = b #"rate of infection"
        self.m = m #"rate of mutation"
        self.p = p #"diffusion weight"
        
        #initial condition array of Susceptible, Infectious, Removed in proportion
        self.s0 = np.ones((M,M)) - i0
        self.i0 = i0
        self.r0 = np.zeros((M,M))

        # Aggregate s0, i0, r0 into one array for use in solve_ivp
        self.sir0 = self.s0.flatten()
        self.sir0 = np.append(self.sir0, self.i0.flatten())
        self.sir0 = np.append(self.sir0, self.r0.flatten())

        # Indices in sir0 that correspond to s, i, r
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
        return "SIRS_PDE(k={0}, b={1}, m={2}, p={3})".format(self.k, self.b, self.m, self.p)
    
    def __repr__(self):
        return self.__str__()
    
    def set_ivp(self, i0, M=200):
        """
        Sets the initial values of this PDE system
        """
        self.s0 = np.ones((M,M)) - i0
        self.i0 = i0
        self.r0 = np.zeros((M,M))
    
    def set_param(self, k, b, m, p):
        """
        Sets the parameters k, b, m, and p of this PDE
        """
        self.k = k
        self.b = b
        self.m = m
        self.p = p

    def simulate(self, teval, figure=False, **kwargs):
        """
        Simulates this PDE for given times in teval. Returns the solution as well as an animation of the simulation if figure=True
        """

        # Simulate
        events = kwargs.get("events", None)
        def f(t, sir):
            sprime = -1*self.b*np.multiply(sir[self.Sind],sir[self.Iind]) + self.p*self.L @ sir[self.Sind] + self.m*sir[self.Rind]
            
            iprime = self.b*np.multiply(sir[self.Sind],sir[self.Iind])-self.k*sir[self.Iind] + self.p*self.L @ sir[self.Iind]

            rprime = self.k*sir[self.Iind] + self.p*self.L @ sir[self.Sind] - self.m*sir[self.Rind]

            return np.append(np.array(sprime), [np.array(iprime), np.array(rprime)])

        solution = solve_ivp(f, t_span=(teval[0], teval[-1]), y0=self.sir0, t_eval=teval, events=events)

        if figure:
            # Animation of infected trajectory
            ifig=plt.figure(figsize=(16,10))
            ifig.set_tight_layout(True)
            ims = []
            for time in np.arange(start=0, stop=teval.shape[0], dtype=int):
                im = plt.imshow(solution.y[self.Iind, time].reshape((self.M, self.M)), animated=True)
                ims.append([im])
                
            plt.axis("off")
            plt.title("Infected - SIRS PDE with k = {0}, b = {1}, m = {2}, p = {3}".format(self.k, self.b, self.m, self.p), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            animi = ArtistAnimation(ifig, ims)

            # Animation of susceptible trajectory
            sfig = plt.figure(figsize=(16,10))
            sfig.set_tight_layout(True)
            sims = []
            for time in np.arange(start=0, stop=teval.shape[0], dtype=int):
                sim = plt.imshow(solution.y[self.Sind, time].reshape((self.M, self.M)), animated=True)
                sims.append([sim])
            
            plt.axis("off")
            plt.title("Susceptible - SIRS PDE with k = {0}, b = {1}, m = {2}, p = {3}".format(self.k, self.b, self.m, self.p), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            anims = ArtistAnimation(sfig, sims)

            # Animation of recovered trajectory
            rfig = plt.figure(figsize=(16,10))
            rfig.set_tight_layout(True)
            rims = []
            for time in np.arange(start=0, stop=teval.shape[0], dtype=int):
                rim = plt.imshow(solution.y[self.Rind, time].reshape((self.M, self.M)), animated=True)
                rims.append([rim])
            
            plt.axis("off")
            plt.title("Recovered - SIRS PDE with k = {0}, b = {1}, m = {2}, p = {3}".format(self.k, self.b, self.m, self.p), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            animr = ArtistAnimation(rfig, rims)

            return (solution, animi, anims, animr)
        else:
            return (solution, None, None, None)

class SIRS_PDE_HETEROGENEOUS(SIRS_PDE):
    """
    A class for the SIRS pde model with different diffusion weights for susceptible, infected, and recovered individuals.
    """

    def __init__(self, k, b, m, ps, pi, pr, i0, M=200):
        """
        Initialize a SIRS_PDE object with parameters governing infection dynamics as well as initial data.
        """
        self.M = M #size of grid (M x M) 
        self.k = k #"rate of recovery"
        self.b = b #"rate of infection"
        self.m = m #"rate of mutation"

        # Diffusion weights
        self.ps = ps
        self.pi = pi
        self.pr = pr
        
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

    def set_ivp(self, i0, M=200):
        """
        Sets the initial values of this PDE system
        """
        self.s0 = np.ones((M,M)) - i0
        self.i0 = i0
        self.r0 = np.zeros((M,M))
        
    def set_param(self, k, b, m, ps, pi, pr):
        """
        Sets the parameters k, b, m, and diffusion weights of this PDE
        """
        self.k = k
        self.b = b
        self.m = m
        self.ps = ps
        self.pi = pi
        self.pr = pr

    def simulate(self, teval, figure=False, **kwargs):
        """
        Simulates this PDE for given times in teval. Returns the solution as well as a plot of the simulation if figure=True
        """

        # Simulate
        events = kwargs.get("events", None)
        def f(t, sir):
            sprime = -1*self.b*np.multiply(sir[self.Sind],sir[self.Iind]) + self.ps*self.L @ sir[self.Sind] + self.m*sir[self.Rind]
            
            iprime = self.b*np.multiply(sir[self.Sind],sir[self.Iind])-self.k*sir[self.Iind] + self.pi*self.L @ sir[self.Iind]

            rprime = self.k*sir[self.Iind] + self.pr*self.L @ sir[self.Sind] - self.m*sir[self.Rind]

            return np.append(np.array(sprime), [np.array(iprime), np.array(rprime)])

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
            plt.title("Infected - SIRS PDE with k = {0}, b = {1}, m = {2}, ps = {3}, pi = {4}, pr = {5}".format(self.k, self.b, self.m, self.ps, self.pi, self.pr), fontsize=20)
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
            plt.title("Susceptible - SIRS PDE with k = {0}, b = {1}, m = {2}, ps = {3}, pi = {4}, pr = {5}".format(self.k, self.b, self.m, self.ps, self.pi, self.pr), fontsize=20)
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
            plt.title("Recovered - SIRS PDE with k = {0}, b = {1}, m = {2}, ps = {3}, pi = {4}, self.pr = {5}".format(self.k, self.b, self.m, self.ps, self.pi, self.pr), fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=15)
            animr = ArtistAnimation(rfig, rims)

            return (solution, animi, anims, animr)
        else:
            return (solution, None, None, None)