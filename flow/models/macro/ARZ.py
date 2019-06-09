import gym
from scipy.optimize import fsolve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def boundary_right(Data):
    # RIGHT boundary condition
    #     RIGHT = Data(:,length(Data))
    return Data[0][len(Data[1]) - 1], Data[1][len(Data[1]) - 1]


def boundary_left(Data):
    # Left boundary condition
    #     LEFT = Data(:,1)
    return Data[0][0], Data[1][0]


################################################
def ARZ_Solve(u_full, u_r, u_l):
    # Variables
    # time and cell step
    step = dt / dx

    # Solve

    # We shouldn't have this in here
    u_l = boundary_left(u_full)
    u_r = boundary_right(u_full)
    # full arrray with boundary conditions
    u_all = np.insert(np.append(u_full[0], u_r[0]), 0, u_l[0]), np.insert(
        np.append(u_full[1], u_r[1]), 0, u_l[1])

    # compute flux
    Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init = Compute_Flux(dt, dx, u_all)

    # update new points
    new_points = ARZ_update_points(Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init, Ve, tau, dt, step)
    return new_points


########################################
# ARZ FLux Functions as given in Shimao Fan paper (page 69)
def Function_rho(density, y_value):
    return y_value + (density * Ve(density))  # Flux equation for density (rho)


def Function_y(density, y_value):
    return ((y_value ** 2) / density) + (y_value * Ve(density))  # Flux equation for (y)


def Compute_Flux(dt, dx, U_all):
    # Method here is inspired by Chapter 4(page 71) of Randal J. Leveque. the
    # pdf text book link is below. We take the Flux definition as the "The Lax?Friedrichs Method"
    # http://www.tevza.org/home/course/modelling-II_2016/books/Leveque%20-%20Finite%20Volume%20Methods%20for%20Hyperbolic%20Equations.pdf

    rho_full = U_all[0];
    y_full = U_all[1];

    # LEFT calculate entire row except last two
    rho_l = rho_full[:-2];
    y_l = y_full[:-2];

    # MID --> init all expect first and last
    rho_init = rho_full[1:-1];
    y_init = y_full[1:-1];

    # RIGHT calculate entire row except first two
    rho_r = rho_full[2:];
    y_r = y_full[2:];
    # LEFT FLUXES
    Flux__rho_L = 0.5 * (Function_rho(rho_l, y_l) + Function_rho(rho_init, y_init)) - (
            (0.5 * dt / dx) * (rho_init - rho_l));
    Flux_y_L = 0.5 * (Function_y(rho_l, y_l) + Function_y(rho_init, y_init)) - ((0.5 * dt / dx) * (y_init - y_l));

    # RIGHT FLUXES
    Flux__rho_R = 0.5 * (Function_rho(rho_r, y_r) + Function_rho(rho_init, y_init)) - (
            (0.5 * dt / dx) * (rho_r - rho_init));
    Flux_y_R = 0.5 * (Function_y(rho_r, y_r) + Function_y(rho_init, y_init)) - ((0.5 * dt / dx) * (y_r - y_init));

    # our final fluxes
    Fp_higher_half = Flux__rho_R
    Fp_lower_half = Flux__rho_L

    Fy_higher_half = Flux_y_R
    Fy_lower_half = Flux_y_L
    return Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init


###########################
def ARZ_update_points(Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init, Ve, tau, dt,step):
    # updating density
    global rho_next
    rho_next = rho_init + (step * (Fp_lower_half - Fp_higher_half))

    # Updating y(relative flow)
    # right hand side constant
    global rhs
    rhs = y_init + (step * (Fy_lower_half - Fy_higher_half)) + ((dt / tau) * ((rho_next * Ve(rho_next))))

    ###Fsolve
    #####update y_value we use F_solve
    x0 = y_init;
    y_next = fsolve(myfun, x0)

    return rho_next, y_next


def myfun(y_next):
    func = y_next + ((dt / tau) * (rho_next * u(rho_next, y_next)) - rhs);
    return func


def Ve(density):
    return u_max * (1 - (density / rho_max))  # Greenshields model for the equilibrium velocity


def u(density, y_value):
    return (y_value / density) + Ve(density)  # velocity function


class ARZ(gym.Env):

    def __init__(self, initial_conditions, boundary_left):
        self.init = initial_conditions
        self.boundary = boundary_left
        self.obs = initial_conditions

    def step(self, rl_actions):
        obs = []
        rew = 0
        done = False
        info_dict = {}

        self.obs = ARZ_Solve(self.obs, rl_actions, self.boundary)

        return obs, rew, done, info_dict

    def reset(self):
        self.obs = self.init

        return self.obs


if __name__ == '__main__':

    global L, N, dx, CFL, dt, x, tau, rho_max, u_max, u_r, u_l
    # define variables, parameters and functions
    # PARAMETERS
    rho_max = 1  # maximum_density
    u_max = 1  # maximum velocity
    #  length of road
    L = 1
    N = 100  # spacial grid resolution /  cell space should be atleast n = 300
    dx = L / N
    # CFL condition--must becloser to or equal to 1 (dictactes the speed of information travel)
    CFL = 0.99
    dt = CFL * dx / u_max
    # scaling -- points on street we are plotting against
    x = np.arange(0.5 * dx, (L - 0.5 * dx), dx)
    # time needed to adjust velocity from u to Ve
    tau = 0.1

    ########
    # initial_data
    # Density
    rho_L_side = 0.5 * (x < max(x) / 2)
    rho_R_side = 0.5 * (x > max(x) / 2)

    # Velocity
    u_L_side = 0.7 * (x < max(x) / 2)
    u_R_side = 0.1 * (x > max(x) / 2)

    u_data_rho_rho = rho_L_side + rho_R_side  # density
    u_data_rho_velocity = u_L_side + u_R_side  # velocity
    # Calculate relative flow
    y_vector = (u_data_rho_rho * (u_data_rho_velocity - Ve(u_data_rho_rho)))



    initial_data = u_data_rho_rho, y_vector
    # Boundary conditions
    u_l = boundary_left(initial_data)
    u_r = boundary_right(initial_data)

    env = ARZ(initial_data, u_l)

    obs = env.reset()

    for _ in range(50):
        action = u_r
        obs, rew, done, _ = env.step(action)
        # density plot
        # our initial data vector = [density ; relative flow]
        plt.plot(x, env.obs[0], 'b-')
        plt.axis([0, L, 0.4, 0.8])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    # final plot
    plt.plot(x, env.obs[0], 'b-')
    plt.axis([0, L, 0.4, 0.8])
    plt.show()
