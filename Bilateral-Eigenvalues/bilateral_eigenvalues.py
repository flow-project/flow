import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt

"""This function is used to compute the eigenvalues of a circular
track in the case where the leading car is using bilateral
control and the type of car alternates manual and
automated.
PARAMETERS:
kd:  spring gain
kv:  velocity damper strength
num_rows: number of rows. Note that the first row is set to
include the boundary condition
OUTPUT: List of the eigenvalues
 """


def circular_track_alternate(kd, kv, num_cars):
    N = np.array([[0., 0], [kd, kv]])
    M = np.array([[0., 1], [-kd, -kv]])

    # add the customary top row
    main_mat = np.zeros((2*num_cars, 2*num_cars))

    main_mat[0:2, :] = np.concatenate((M, N/2, np.zeros((2, 2*num_cars - 6)),
                                       N/2), axis=1)
    # Used to store the files we will be concatenating
    for i in range(1, num_cars):
        temp_row = 0
        if i % 2 == 1:
            # Create the row that we will insert
            temp_row = np.concatenate((N/2., M, np.zeros((2, 2*num_cars - 4))),
                                      axis=1)
        else:
            temp_row = np.concatenate((N/2, M, N/2,
                                      np.zeros((2, 2*num_cars - 6))), axis=1)

        temp_row = np.roll(temp_row, 2*(i-1))

        main_mat[2*i:2*(i+1), :] = temp_row

    e_vals, e_vecs = LA.eig(main_mat)
    return e_vals


"""This function is used to compute the eigenvalues of a
circular track in the case where all cars are using a non-delayed
car following model

PARAMETERS:
kd:  spring gain
kv:  velocity damper strength
num_rows: number of rows. Note that the first row is set to
include the boundary condition
OUTPUT: List of eigenvalues
 """


def circular_track_manual(kd, kv, num_cars):
    N = np.array([[0., 0], [kd, kv]])
    M = np.array([[0., 1], [-kd, -kv]])

    # add the customary top row
    main_mat = np.zeros((2*num_cars, 2*num_cars))

    main_mat[0:2, :] = np.concatenate((M, np.zeros((2, 2*num_cars - 4)),
                                      N), axis=1)
    # Used to store the files we will be concatenating
    for i in range(1, num_cars):

        temp_row = np.concatenate((N, M, np.zeros((2, 2*num_cars - 4))),
                                  axis=1)
        temp_row = np.roll(temp_row, 2*(i-1))
        main_mat[2*i:2*(i+1), :] = temp_row

    e_vals, e_vecs = LA.eig(main_mat)
    return e_vals


"""This function is used to compute the eigenvalues of a circular track
in the case where the leading car is using bilateral control and the type
of car after that is manual with probability p_manual and automated with
probability (1-p_manual)
PARAMETERS:
kd:  spring gain
kv:  velocity damper strength
p_manual: the probability that a given car will be manual
num_rows: number of rows. Note that the first row is set to include the
boundary condition
RETURN: List of eigenvalues
 """


def circular_track_alternate_random(kd, kv, p_manual, num_cars):
    N = np.array([[0., 0], [kd, kv]])
    M = np.array([[0., 1], [-kd, -kv]])

    # add the customary top row
    main_mat = np.zeros((2*num_cars, 2*num_cars))

    main_mat[0:2, :] = np.concatenate((M, N/2, np.zeros((2, 2*num_cars - 6)),
                                      N/2), axis=1)
    # Used to store the files we will be concatenating
    for i in range(1, num_cars):
        temp_row = 0
        # generate a random number between zero and 1
        rand_num = np.random.uniform()
        if rand_num > p_manual:
            # Create the row that we will insert
            temp_row = np.concatenate((N/2., M, np.zeros((2, 2*num_cars - 4))),
                                      axis=1)
        else:
            temp_row = np.concatenate((N/2, M, N/2,
                                      np.zeros((2, 2*num_cars - 6))), axis=1)

        temp_row = np.roll(temp_row, 2*(i-1))

        main_mat[2*i:2*(i+1), :] = temp_row

    e_vals, e_vecs = LA.eig(main_mat)
    return e_vals


""" This function is used to calculate the maximum eigenvalue of the system
    over a range of kd and kv for the case where cars alternate manual
    and automated. It also saves a file indicating the areas
    where the system is platoon stable
    PARAMETERS: kd_min: min value of distance gain k_d
                kd_max: max value of distance gain k_d
                kv_min: min value of velocity gain k_v
                kv_max: max value of velocity gain k_v
                spacing: discretization step for k_d, k_v
                num_cars: total number of cars to consider
    OUTPUT: saved file containing zero where system is unstable
    1, where stable. """


def save_eigenvals_alternate(kd_min, kd_max, kv_min, kv_max,
                             spacing, num_cars):
    kd_space = np.linspace(kd_min, kd_max, spacing)
    kv_space = np.linspace(kv_min, kv_max, spacing)

    is_stable = np.zeros((kd_space.shape[0], kv_space.shape[0]))

    for i in xrange(kd_space.shape[0]):
        print i
        for j in xrange(kv_space.shape[0]):
            eig = circular_track_alternate(kd_space[i], kv_space[j],
                                           num_cars)
            eig_real = eig.real
            max_val = eig_real.max()
            if max_val > 0:
                is_stable[i, j] = 0
            else:
                is_stable[i, j] = 1

    # Now we save is_stable for later plotting
    np.savetxt('alternate_stability.out', is_stable, delimiter=',')


""" This function is used to calculate the maximum eigenvalue of the
    system over a range of kd and kv for the case where the probability
    of a car being manual is given by p_manual.
    It also saves a file containing
    the areas where the system is platoon stable
    PARAMETERS: kd_min: min value of distance gain k_d
                kd_max: max value of distance gain k_d
                kv_min: min value of velocity gain k_v
                kv_max: max value of velocity gain k_v
                p_manual: probability that a given car is manual
                spacing: discretization step for k_d, k_v
                num_cars: total number of cars to consider
    OUTPUT: saved file containing zero where system is unstable
    1, where stable."""


def save_eigenvals_alternate_random(kd_min, kd_max, kv_min, kv_max,
                                    p_manual, spacing, num_cars):
    kd_space = np.linspace(kd_min, kd_max, spacing)
    kv_space = np.linspace(kv_min, kv_max, spacing)

    is_stable = np.zeros((kd_space.shape[0], kv_space.shape[0]))

    for i in xrange(kd_space.shape[0]):
        print i
        for j in xrange(kv_space.shape[0]):
            eig = circular_track_alternate_random(kd_space[i], kv_space[j],
                                                  p_manual, num_cars)
            eig_real = eig.real
            max_val = eig_real.max()
            if max_val > 0:
                is_stable[i, j] = 0
            else:
                is_stable[i, j] = 1

    # Now we save is_stable for later plotting
    np.savetxt('alternate_stability_random.out', is_stable, delimiter=',')


""" This function is used to calculate the maximum eigenvalue of the
    system over a range of kd and kv for the case where all cars
    are manual. It also saves a file containing
    the areas where the system is platoon stable
    PARAMETERS: kd_min: min value of distance gain k_d
                kd_max: max value of distance gain k_d
                kv_min: min value of velocity gain k_v
                kv_max: max value of velocity gain k_v
                spacing: discretization step for k_d, k_v
                num_cars: total number of cars to consider
    OUTPUT: saved file containing zero where system is unstable
    1, where stable. """


def save_eigenvals_manual(kd_min, kd_max, kv_min, kv_max, spacing, num_cars):
    kd_space = np.linspace(kd_min, kd_max, spacing)
    kv_space = np.linspace(kv_min, kv_max, spacing)

    is_stable = np.zeros((kd_space.shape[0], kv_space.shape[0]))

    for i in xrange(kd_space.shape[0]):
        for j in xrange(kv_space.shape[0]):
            eig = circular_track_manual(kd_space[i], kv_space[j], num_cars)
            eig_real = eig.real
            max_val = eig_real.max()
            if max_val > 0:
                is_stable[i, j] = 0
            else:
                is_stable[i, j] = 1

    # Now we save is_stable for later plotting
    np.savetxt('manual_stability.out', is_stable, delimiter=',')


if __name__ == '__main__':
    circular_track_alternate(.2, .1, 3)
    # circular_track_manual(2,.2,3)
    # save_eigenvals_alternate(0, .3, 0, .3, 100, 100)
    # save_eigenvals_manual(0, .3, 0, .3, 100, 10)
    # save_eigenvals_alternate_random(0, .3, 0, .3, .9, 100, 50)
