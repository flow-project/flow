from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


##########################
def boundary_right(Data):
    # RIGHT boundary condition
    #     RIGHT = Data(:,length(Data))
    return Data[0][len(Data[1]) - 1], Data[1][len(Data[1]) - 1]


def boundary_left(Data):
    # Left boundary condition
    #     LEFT = Data(:,1)
    return Data[0][0], Data[1][0]


################################################
def compute_next_points(U_Data_full, Ve, u, tau, dt, dx, step):
    # Boundary conditions
    Constant_Left_rho, Constant_Left_y = boundary_left(U_Data_full);
    Constant_Right_rho, Constant_Right_y = boundary_right(U_Data_full);

    # full arrray with boundary conditions
    U_all = np.insert(np.append(U_Data_full[0], Constant_Right_rho), 0, Constant_Left_rho), np.insert(
        np.append(U_Data_full[1], Constant_Right_y), 0, Constant_Left_y)

    #     if length(U_all) > 2:
    # compute flux
    Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init = Compute_Flux(dt, dx, U_all, Ve)

    # update new points
    new_points = ARZSolve(Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init, Ve, u, tau,
                          dt, dx, step)

    return new_points


########################################
# ARZ FLux Functions as given in Shimao Fan paper (page 69)
def Function_rho(density, y_value):
    return y_value + (density * Ve(density))  # Flux equation for density (rho)


def Function_y(density, y_value):
    return ((y_value ** 2) / density) + (y_value * Ve(density))  # Flux equation for (y)


def Compute_Flux(dt, dx, U_all, Ve):
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
    Fp_higher_half = Flux__rho_R;
    Fp_lower_half = Flux__rho_L;

    Fy_higher_half = Flux_y_R;
    Fy_lower_half = Flux_y_L;
    return Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init


###########################
def ARZSolve(Fp_higher_half, Fp_lower_half, Fy_higher_half, Fy_lower_half, rho_init, y_init, Ve, u, tau, dt, dx, step):
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

# define variables, parameters and functions

# PARAMETERS

rho_max = 1  # maximum_density
u_max = 1  # maximum velocity


def Ve(density):
    return u_max * (1 - (density / rho_max))  # Greenshields model for the equilibrium velocity


def u(density, y_value):
    return (y_value / density) + Ve(density)  # velocity function


# Variables
L = 1  # length of road

N = 100  # spacial grid resolution /  cell space should be atleast n = 300
dx = L / N

# CFL condition--must becloser to or equal to 1 (dictactes the speed of information travel)
CFL = 0.96
global dt
dt = CFL * dx / u_max

# scaling -- points on street we are plotting against
x = np.arange(0.5 * dx, (L - 0.5 * dx), dx)
# time needed to adjust velocity from u to Ve
global tau
tau = 0.1
# time and cell step
step = dt / dx

# initial_data
# Density
rho_L_side = 0.5 * (x < max(x) / 2);
rho_R_side = 0.5 * (x > max(x) / 2);

# Velocity
u_L_side = 0.7 * (x < max(x) / 2);
u_R_side = 0.1 * (x > max(x) / 2);

U_Data_rho = rho_L_side + rho_R_side;  # density
U_Data_velocity = u_L_side + u_R_side;  # velocity

# Calculate relative flow
y_vector = (U_Data_rho * (U_Data_velocity - Ve(U_Data_rho)))

#plt.ion()
data_points = U_Data_rho, y_vector  # our initial data vector = [density ; relative flow]
plt.plot(x, data_points[0], 'b-')
plt.axis([0, L, 0.4, 0.8])
plt.draw()
plt.clf()
plt.pause(0.0001)

num_of_iterations = np.arange(100)
for i in num_of_iterations:
    # Solve
    new_points = compute_next_points(data_points, Ve, u, tau, dt, dx, step);
    data_points = new_points
    # density plot
    # our initial data vector = [density ; relative flow]
    plt.plot(x, data_points[0], 'b-')
    plt.axis([0, L, 0.4, 0.8])
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

# final plot
plt.plot(x, data_points[0], 'b-')
plt.axis([0, L, 0.4, 0.8])
plt.show()