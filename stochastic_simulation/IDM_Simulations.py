import numpy as np
from scipy.integrate import odeint
import time
import pickle

import warnings
warnings.filterwarnings('ignore')
import pdb


def prob_enter(headway, vel, density, beta=.9825):
    """Probability of a car entering any of the gaps inbetween vehicles

    :returns: boolean array, True if a car enters a gap
                             False if a car does not enter the gap
    """

    th = headway  # np.divide(headway,vel)  # time headway

    mu_lc = 3.00
    sigma_lc = .3642

    mu_th = 2.9512
    sigma_th = 0.4255

    # normalization constant
    C = np.sqrt(1 / sigma_lc ** 2 - 1 / sigma_th ** 2) \
        * np.exp(
        ((mu_lc - mu_th) ** 2 - 2 * mu_th * sigma_lc ** 2 + 2 * mu_lc * sigma_th ** 2 + sigma_th ** 2 * sigma_lc ** 2) /
        (-2 * sigma_th ** 2 + 2 * sigma_lc ** 2)) \
        / np.sqrt(2 * np.pi)

    p_appear = C * np.exp(
        (np.log(th) - mu_th) ** 2 / (2 * sigma_th ** 2) - (np.log(th) - mu_lc) ** 2 / (2 * sigma_lc ** 2))

    return np.random.rand(len(headway)) < p_appear


def prob_exit(headway, vel, density, beta=.9825):
    """Probability of cars in the lane exiting

    :returns: boolean array, True if a car exits the lane
                             False if a car does not exit the lane
    """

    th = headway  # np.divide(headway,vel)  # time headway

    mu_lc = 2.7962
    sigma_lc = 0.4176

    mu_th = 2.9512
    sigma_th = 0.4255

    # normalization constant
    C = np.sqrt(1 / sigma_lc ** 2 - 1 / sigma_th ** 2) \
        * np.exp(
        ((mu_lc - mu_th) ** 2 - 2 * mu_th * sigma_lc ** 2 + 2 * mu_lc * sigma_th ** 2 + sigma_th ** 2 * sigma_lc ** 2) /
        (-2 * sigma_th ** 2 + 2 * sigma_lc ** 2)) \
        / np.sqrt(2 * np.pi)

    p_disappear = C * np.exp(
        (np.log(th) - mu_th) ** 2 / (2 * sigma_th ** 2) - (np.log(th) - mu_lc) ** 2 / (2 * sigma_lc ** 2))

    if len(headway) <= 4:
        return [False]*len(headway)
    else:
        return np.random.rand(len(headway)) < p_disappear


def car_following(y, t, params):
    """Defines car-following behavior of driver-vehicle units.
       Behvaior here is specified by the Improved Intelligent Driver Model (IIDM)

    :y: state of the system
        - first half of the array represents the position of the cars, ordered in decreasing position
        - second half of the array represents the velocity of the cars, ordered as they are in the position half
    :t: time interval
    :params : parameters of the IDM car-following model
              - params['v0']    : desirable velocity [m/s]
              - params['T']     : safe time headway [s]
              - params['a']     : maximum acceleration [m/s2]
              - params['b']     : comfortable decceleration [m/s2]
              - params['delta'] : acceleration exponent [unitless]
              - params['lc']    : average vehicle length [m]
              - params['s0']    : linear jam distance [m]
              - params['s1']    : nonlinear jam distance [m]
              - params['lr']    : length of the circular road [m]

    :returns: dydt - derivative of the states (y)
    """

    dydt = np.zeros(len(y))
    n = int(len(y)/2)

    # headway_cur = np.zeros(n)  # headway at current time step
    # v_cur = y[n:]              # velocity at current time step

    for i in range(n):
        if i == 0:
            s = y[n - 1] - y[0] - params['lc'] + params['lr']  # bumper-to-bumper gap
            vl = y[2 * n - 1]  # velocity of lead vehicle
        else:
            s = y[i - 1] - y[i] - params['lc']
            vl = y[n + i - 1]

        v = y[n + i]  # velocity of current vehicle

        s_star = params['s0'] + max([0, v * params['T'] + v * (v - vl) / (s * np.sqrt(params['a'] * params['b']))])

        dydt[i] = v
        dydt[n + i] = params['a'] * (1 - (v / params['v0']) ** params['delta'] - (s_star / s) ** 2)

        # headway_cur[i] = s

    # # failsafe velocity
    # max_deacc = 5
    # d = -headway_cur - np.square(np.append(v_cur[-1:],v_cur[:-1]))/(2*max_deacc)
    # v_failsafe = -max_deacc*tau + np.sqrt(max_deacc)*np.sqrt(-2*d + max_deacc*tau**2)
    #
    # # failsafe acceleration
    # a_failsafe = (v_failsafe-v_cur)/dt
    #
    # # implement failsafe
    # ind_notsafe = a_failsafe<dydt[n:]
    # dydt[n:][ind_notsafe] = a_failsafe[ind_notsafe]
    # dydt[:n][ind_notsafe] = v_failsafe[ind_notsafe]

    return dydt


class RingRoad:
    """
    Simulation of vehicles in a circular lane following the IDM model.
    Vehicles are allowed to enter and exit the lane according to the probability "prob_enter" and "prob_exit"
    Certain vehicle units of your choosing can be held constant in the model for analysis purposes
    """

    def __init__(self, params, n_cars, x_init, v_init, ind_cars_const):
        """Instantiates the class with the car-following model parameters
           Initializes road with a set of vehicles with inital positions and velocities
           Specifies vehicles that should not be allowed to exit the road (may be equal to zero)

        :params         : parameters of the car-following model, MUST CONTAIN lr and lc
                           - params['lc'] : average vehicle length [m]
                           - params['lr'] : length of the circular road [m]
        :n_cars         : initial number of cars on the road [unitless]
        :x_init         : inital position of each car in the set [m]
        :v_init         : initial velocity of each car in the set [m/s]
        :ind_cars_const : index of cars in the set that should be held constant (length equal to n_cars_const)
        """
        # car-following parameters
        self.params = params

        # parameters that are crucial for simulation
        self.lc = params['lc']
        self.lr = params['lr']
        self.tau = params['tau']

        # initial conditions of vehicles
        self.n_cars = n_cars
        self.x_init = x_init
        self.v_init = v_init
        self.h_init = np.append(-x_init[-1] + x_init[0] + params['lr'], x_init[1:] - x_init[:-1]) - params['lc']

        # cars that cannot exit the road
        self.ind_cars_const = ind_cars_const

    def simulate(self, dt, t_final, t_lc=0):
        headway_disappear = np.array([])
        gap_appear = np.array([])

        tlc = min(t_lc, dt)
        n_cars_const = len(self.ind_cars_const)

        # store constant cars in the first n columns
        ind_variable_cars = np.arange(self.n_cars)
        ind_variable_cars = ind_variable_cars[np.invert(np.in1d(ind_variable_cars, self.ind_cars_const))]
        ind = np.append(self.ind_cars_const, ind_variable_cars).astype(int)

        # reorganize initial conditions given new index arrangement
        y0 = np.append(self.x_init[ind], self.v_init[ind])

        # time range from 0 to t_final
        t = np.arange(0, t_final + dt, dt)

        # initialize variables of interest
        headway = np.zeros((len(t), int(len(y0) / 2)))
        headway[0, :] = self.h_init[ind]
        sol = np.zeros((len(t), len(y0)))
        sol[0, :] = y0
        n_cars_tot = self.n_cars  # total number of cars to be in the lane
        n_cars_cur = np.append(self.n_cars, np.zeros(len(t)-1))  # current number of cars in the lane
        ind_cars = np.arange(self.n_cars)
        ind_cars = ind_cars[np.argsort(y0[:int(len(y0)/2)][::-1])]  # indeces of the cars currently in the lane
        # organized in decreasing order of position
        num_exits = np.zeros(len(t))
        num_enters = np.zeros(len(t))

        for i in range(int(t_final/t_lc)):

            # calculate next position and velocity of current cars in lane

            t_min = int(i * t_lc / dt) + 1  # index corresponding to most recent lane change
            t_max = int((i + 1) * t_lc / dt) + 1  # index corresponding to next lane change

            sol_i = odeint(car_following, sol[t_min - 1, np.append(ind_cars, ind_cars + n_cars_tot)],
                           np.arange(0, t_lc + dt, dt), args=(self.params,))

            # store collected data on position, velocity, and headway

            k = 1
            for j in np.arange(t_min, t_max):
                # store new position and velocity data
                sol[j, np.append(ind_cars, ind_cars + n_cars_tot)] = sol_i[k, :]
                k += 1

                # store new headway data
                headway[j, ind_cars] = np.append(sol[j, ind_cars[-1]] - sol[j, ind_cars[0]] + self.lr,
                                                 sol[j, ind_cars[:-1]] - sol[j, ind_cars[1:]])

                n_cars_cur[j] = sum(sol[j, :] != 0)/2

            # check for situations of lane changes given velocity and headway

            # headway of each car currently in the lane
            headway_cur = np.append(sol[t_max - 1, ind_cars[-1]] - sol[t_max - 1, ind_cars[0]] + self.lr,
                                    sol[t_max - 1, ind_cars[:-1]] - sol[t_max - 1, ind_cars[1:]])

            # velocity of each car currently in the lane
            vel_cur = sol[t_max - 1, ind_cars + n_cars_tot]

            enter = prob_enter(headway_cur, vel_cur, dt)  # determine which gaps accept new cars
            exit = prob_exit(headway_cur, vel_cur, dt)  # determine which cars exit the lane
            # ensure that cars that are constant do not exit the lane
            exit = np.logical_and(exit, np.logical_not(np.in1d(ind_cars, np.arange(n_cars_const))))

            # adjust headways of after vehicles exit
            if sum(exit) > 0:
                headway_disappear = np.append(headway_disappear, headway_cur[exit])

            #     ind_exit = np.where(exit)[0]
            #
            #     # add headways to lagging vehicle of exited vehicles
            #     for j in range(len(ind_exit)):
            #         if ind_cars[ind_exit[j]] == ind_cars[-1]:
            #             headway[t_max-1, ind_cars[0]] += headway[t_max-1, ind_cars[-1]]
            #             headway[t_max-1, ind_cars[-1]] = 0
            #         else:
            #             headway[t_max-1, ind_cars[ind_exit[j]+1]] += headway[t_max-1, ind_cars[ind_exit[j]]]
            #             headway[t_max-1, ind_cars[ind_exit[j]]] = 0

            # update variables given cars that enter and/or exit
            if sum(enter) > 0:
                gap_appear = np.append(gap_appear, headway_cur[enter])
                ind = np.where(enter)[0]

                # calculate the position of the new vehicle (halfway point of the gap)
                if enter[0]:
                    x_new = np.append([1/2 * (sol[t_max-1, ind_cars[-1]] - sol[t_max-1, ind_cars[0]] + self.lr) +
                                      sol[t_max - 1, ind_cars[0]]], [1/2 * (sol[t_max-1, ind_cars[ind[1:]]] +
                                                                            sol[t_max-1, ind_cars[ind[1:]-1]])])
                else:
                    x_new = 1/2 * (sol[t_max-1, ind_cars[ind]] + sol[t_max - 1, ind_cars[ind - 1]])

                # calculate the velocity of the new vehicle (velocity of lagging vehicle)
                v_new = sol[t_max-1, ind_cars[ind]+n_cars_tot]

                # calculate the headway of the new vehicle and the vehicle behind it
                # h_new = 0.5 * headway[t_max-1, ind_cars[ind]]
                # h_lag_new = 0.5 * headway[t_max-1, ind_cars[ind]]

                # add columns to sol to compensate for the presence of new vehicles
                sol = np.insert(sol, [n_cars_const], np.zeros((sol.shape[0], sum(enter))), axis=1)
                sol = np.insert(sol, [sum(enter) + n_cars_tot + n_cars_const], np.zeros((sol.shape[0], sum(enter))),
                                axis=1)

                # add initial data of new vehicles into sol
                sol[t_max - 1, np.arange(sum(enter)) + n_cars_const] = x_new
                sol[t_max - 1, np.arange(sum(enter)) + n_cars_tot + sum(enter) + n_cars_const] = v_new

                # add columns to headway to compensate for the presence of new vehicles
                headway = np.insert(headway, [n_cars_const], np.zeros((headway.shape[0], sum(enter))), axis=1)

                # update data in headway matrix to account for changes
                # headway[t_max - 1, np.arange(sum(enter)) + n_cars_const] = h_new
                # headway[t_max - 1, ind_cars[ind]] = h_lag_new

            # update indices (whether change occurred or not), and add cars (if change occured)
            exit = np.logical_or(exit, np.in1d(ind_cars, np.arange(n_cars_const)))
            ind_cars = np.append(np.append(np.arange(n_cars_const), np.arange(sum(enter)) + n_cars_const),
                                 ind_cars[np.invert(exit)] + sum(enter))
            # order the indices by position
            ind_cars = ind_cars[np.argsort(sol[t_max - 1, ind_cars])[::-1]]

            n_cars_tot += sum(enter)
            num_exits[t_max - 1] = sum(exit)
            num_enters[t_max - 1] = sum(enter)

        pos_rad = np.divide(sol[:, :int(sol.shape[1] / 2)], self.lr) * 2 * np.pi  # position in terms of radians
        pos_absolute = sol[:, :int(sol.shape[1] / 2)]  # absolute position of every car, in meters
        vel = sol[:, int(sol.shape[1] / 2):]  # velocity of every car at every point in time, 0 if car is not available

        # print('Mean disappearing headway:', np.mean(headway_disappear))
        # print('Std disappearing headway:', np.std(headway_disappear))
        # print('Mean appearing gap:', np.mean(gap_appear))
        # print('Std appearing gap:', np.std(gap_appear))

        return pos_absolute, pos_rad, vel, headway, n_cars_cur, num_exits, num_enters


if __name__ == '__main__':
    # define parameters of interest for the car following model
    params = {'v0': 30, 'T': 1.5, 'a': 1, 'b': 3, 'delta': 4, 'lc': 0, 's0': 2, 's1': 3, 'lr': 230, 'tau': 0.4, }

    # specify initial conditions
    n_cars = 11  # number of cars
    x_init = t = np.linspace(0, float(params['lr']) - float(params['lr']) / n_cars, n_cars)  # initial position of cars
    v_init = 10 * np.ones(n_cars)  # initial velocity of cars

    # indices of cars that do not change lanes
    ind_cars_const = np.array([])


    # simulation parameters
    dt = 0.025      # update time [s]
    t_final = 2000  # simulation time [s]
    #lane_change_step = 8  # must be a multiple of dt
    lane_change_step = 2000
    num_simulations = 1  # number of simulations to perform
    show_statistics = True
    export_data = True

    if lane_change_step > 100:
        x_init = x_init + (6*np.random.rand(n_cars) - 3)

    # initialize model
    model = RingRoad(params, n_cars, x_init, v_init, ind_cars_const)

    avg_car_num = np.zeros(num_simulations)
    avg_vel = np.zeros(num_simulations)
    avg_lc = np.zeros(num_simulations)

    # begin simulation
    simulation_output = []
    t_begin = time.time()  # initial time for all simulations
    for simulation_num in range(num_simulations):

        print('Beginning Simulation %d...' % (simulation_num+1))

        t1 = time.time()  # initial time

        # performs ith simulation
        pos_absolute, pos_rad, vel, headway, n_cars_cur, num_exits, num_enters = \
            model.simulate(dt, t_final, t_lc=lane_change_step)

        # store data from simulation
        simulation_output.append({'position': pos_absolute, 'velocity': vel, 'headway': headway,
                                  'num_cars': n_cars_cur, 'num_enters': num_enters, 'num_exits': num_exits})

        if show_statistics:
            avg_car_num[simulation_num] = np.average(n_cars_cur)
            avg_vel[simulation_num] = np.average(np.sum(vel, axis=0) / np.sum(vel != 0, axis=0))
            avg_lc[simulation_num] = np.sum(num_exits) + np.sum(num_enters)

        t2 = time.time()  # final time

        print('Done! Simulation Time: %.2fs' % (t2-t1))
        print('--------------------------------')

    if show_statistics:
        print('Simulation Statistics:')
        print(" - Average velocity: %.2f" % np.average(avg_vel))
        print(" - Average number of vehicles: %.2f" % np.average(avg_car_num))
        print(" - Average number of lane changes: %.2f" % np.average(avg_lc))
        print('--------------------------------')

    if export_data:
        print('Exporting data...')

        # export collected data in a pickle file
        output_filename = 'data.pkl'
        output = open(output_filename, 'wb')
        pickle.dump(simulation_output, output)
        output.close()

        print('Data exported to: %s' % output_filename)
        print('--------------------------------')

    t_terminate = time.time()  # final time for all simulations

    print('Simulations complete! Total run time: %.2fs' % (t_terminate-t_begin))
