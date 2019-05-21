import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def initial(x):
    values = 1 * (x <= 5) + (-4 + x) * (x > 5) * (x <= 6) + 2 * (x > 6) * (x < 15) + (2 * x - 28) * (x > 15) * (
                x <= 16) + 4 * (x > 16) * (x < 25) + 1 * (x >= 25)
    return values

def Uleft(T):
    Ul = 0
    return Ul

def Uright(T):
    Ur = 0
    return Ur

def Gflux(U, V, R):
    #demand
    D = V * U * (1 - U / R) * (U < 0.5 * R) + 0.25 * V * R * (U >= 0.5 * R)
    #supply
    S = V * U * (1 - U / R) * (U > 0.5 * R) + 0.25 * V * R * (U <= 0.5 * R)
    S = np.append(S[1:], S[len(S) - 1])
    #Godunov flux
    F = np.minimum(D, S)
    return F

def IBVP(U, U_R, U_L):
    # Godunov scheme for multi-populations
    # P.Goatin, June 2017
    # clf
    # """"parameters"""
    L = 30  # length of road
    N = 0.5  # (N = L/dx)reduce spacial grid resolution
    dx = L / N

    x = np.array([1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5])

    V = 1
    R = 4

    CFL = 0.95
    dt = CFL * dx / V
    lam = dt / dx

    U = np.insert(np.append(U, U_R), 0, U_L)

    # Godunov numerical flux

    F = Gflux(U, V, R)

    Fm = np.insert(F[0:len(F) - 1], 0, F[0])

    # Godunov scheme  (UPDATING U)
    U = U - lam * (F - Fm)

    U = np.insert(np.append(U[1:len(U) - 1], U_R), 0, U_L)

    # plot current profile during execution

    plt.plot(x, U[1:len(U) - 1], 'b-')
    plt.axis([0, L, -0.1, 4.1])
    plt.show()

    return U[1:len(U)-1]


class LWR(gym.Env):

    def __init__(self, initial_conditions, boundary_left):
        self.init = initial_conditions
        self.boundary = boundary_left
        self.obs = initial_conditions

    def step(self, rl_actions):
        """

        Parameters
        ----------
        rl_actions : int or array_like
            actions to be performed by the agent

        Returns
        -------
        array_like
            next observation
        float
            reward
        bool
            done mask
        dict
            additional information (defaults to {})
        """
        obs = []
        rew = 0
        done = False
        info_dict = {}

        u_r = rl_actions
        # TODO: advance the state of the simulation by one step
        self.obs = IBVP(self.obs, u_r, self.boundary)

        return obs, rew, done, info_dict


    def reset(self):
        """

        :return:
        """
        self.obs = self.init

        return self.obs

# a few more parameters
# Length of road

x = np.array([1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5]) # points on x axis to plot
U = initial(x) #compute initial points
U_R = Uright(0) #right boundary
U_L = Uleft(0) #left boundary

if __name__ == "__main__":
   env = LWR(U, U_L)

   obs = env.reset()
   for _ in range(10):
       action = None  # agent.compute(obs)
       obs, rew, done, _ = env.step(U_R)
       # update plot
