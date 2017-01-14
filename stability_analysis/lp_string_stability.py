import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from control import tf, pade, bode_plot, tfdata, impulse_response

# Constants
THETA_MIN = 0
THETA_MAX = 1.5

TAU = 0.1 # seconds
KP = 0.2
KD = 0.7
KDD = 0
H = 0.5 # seconds
THETAS = [0, 0.15, 0.3]

order = 5  # pade approximation order


def construct_system_tf(h=0.0, theta=0.0, kp=0.0, kd=0.0, kdd=0.0, tau=0.0):
    # Coefficients in denominator of transfer function
    # High order to low order, eg 1*s^2 + 0.1*s + 1
    den = [h * tau, tau + h * (1 + kdd), kdd + 1 + h * kd, kd + h * kp, kp]

    # non-delayed part of the system
    num1 = [kdd, kd, kp]
    sys1 = tf(num1, den)

    # delayed part of the system, without the delay
    num_nodelay = [tau, 1, 0, 0]
    sys_nodelay = tf(num_nodelay, den)

    # add in the delay to the delayed part of the system
    num_delay, den_delay = pade(theta, n=order)
    sys_delay = tf(num_delay, den_delay)
    sys2 = sys_nodelay * sys_delay

    # combine the delayed and nondelayed parts
    sys = sys1 + sys2

    return sys


def construct_system_block(h=0.0, theta=0.0, kp=0.0, kd=0.0, kdd=0.0, tau=0.0):
    # See Ploeg2014 for the notation and variable naming used here. "inv" indicates inverse.
    Hinv = tf([1], [h, 1])
    K = tf([kdd, kd, kp], [1])
    G = tf([1], [tau, 1, 0, 0])
    KG = K * G
    num_delay, den_delay = pade(theta, n=order)
    D = tf(num_delay, den_delay)
    one = tf([1], [1])
    num_KG, den_KG = tfdata(KG + one)
    invKG1 = tf(den_KG, num_KG)
    sys = invKG1 * (KG + D) * Hinv

    return sys

if __name__ == "__main__":
    print "Reproducing Figure 3(a) from Ploeg2014 (1): reduced transfer function"
    fig = plt.figure()
    for theta in THETAS:
        sys = construct_system_tf(h=H, theta=theta, kp=KP, kd=KD, kdd=KDD, tau=TAU)
        mag1, phase1, omega1 = bode_plot(sys, dB=True, deg=False, Plot=True)

    fig.axes[0].set_ylim(bottom=-5, top=1)
    fig.axes[0].set_xlim(left=0.05, right=10)
    fig.axes[1].set_xlim(left=0.05, right=10)
    plt.suptitle("Reproducing Figure 3(a) [Ploeg2014]")
    plt.savefig("lp_string_stability_3a_tf_mag.png", dpi=300, format="png")


    print "Reproducing Figure 3(a) from Ploeg2014 (2): using block diagram algebra"
    fig = plt.figure()
    for theta in THETAS:
        sys = construct_system_block(h=H, theta=theta, kp=KP, kd=KD, kdd=KDD, tau=TAU)
        mag2, phase2, omega2 = bode_plot(sys, dB=True, deg=False, Plot=True)

    fig.axes[0].set_ylim(bottom=-5, top=1)
    fig.axes[0].set_xlim(left=0.05, right=10)
    fig.axes[1].set_xlim(left=0.05, right=10)
    plt.suptitle("Reproducing Figure 3(a) [Ploeg2014]")
    plt.savefig("lp_string_stability_3a_block_mag.png", dpi=300, deg=False, format="png")

    assert np.linalg.norm(mag1 - mag2) < 1e-8, "Magnitudes produced by the two methods are different"
    assert np.linalg.norm(omega1 - omega2) < 1e-8, "Frequency list produced by the two methods are different"

    # FIXME why are the phases produced by the two attempts different?
    # assert np.linalg.norm(phase1 - phase2) < 1e-8, "Phases produced by the two methods are different"


    print "Reproducing Figure 3(b) from Ploeg2014"
    h_range = np.linspace(0, 1.5, 20)
    theta_range = np.zeros(h_range.shape)

    for i, h in enumerate(h_range):
        # binary search for theta
        theta_min = THETA_MIN
        theta_max = THETA_MAX
        while abs(theta_min-theta_max) > 1e-3:
            theta = 0.5 * (theta_min + theta_max)
            sys = construct_system_block(h=h, theta=theta, kp=KP, kd=KD, kdd=KDD, tau=TAU)
            mag, phase, omega = bode_plot(sys, dB=True)
            Hinfty = np.max(mag)  # ||Gamma(s)||_{H_\infty}
            if Hinfty >= 1:
                theta_max = theta
            else:
                theta_min = theta
        theta_range[i] = theta_min

    fig = plt.figure()
    plt.plot(h_range, theta_range)
    plt.xlabel("h [s]")
    plt.ylabel("theta_max [s]")
    plt.suptitle("Reproducing Figure 3(b) [Ploeg2014]")
    plt.savefig("lp_string_stability_3b_theta_max.png", dpi=300, deg=False, format="png")


    print "Reproducing Figure 4(a) from Ploeg2014"

    fig = plt.figure()
    for theta in THETAS:
        sys = construct_system_block(h=H, theta=theta, kp=KP, kd=KD, kdd=KDD, tau=TAU)
        T, yout = impulse_response(sys)
        plt.plot(T, yout, label='theta=%s' % theta)

        L1 = np.trapz(yout, T)  # ||gamma(t)||_L1

    fig.axes[0].set_ylim(bottom=0, top=2.2)
    fig.axes[0].set_xlim(left=0, right=5)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("gamma [m/s^2]")
    plt.suptitle("Reproducing Figure 4(a) [Ploeg2014]")
    plt.savefig("lp_string_stability_4a_impulse.png", dpi=300, deg=False, format="png")


    print "Reproducing Figure 4(b) from Ploeg2014"
    h_range = np.linspace(0, 1.5, 20)
    theta_range = np.zeros(h_range.shape)

    for i, h in enumerate(h_range):
        # binary search for theta
        theta_min = THETA_MIN
        theta_max = THETA_MAX
        while abs(theta_min-theta_max) > 1e-3:
            theta = 0.5 * (theta_min + theta_max)
            sys = construct_system_block(h=h, theta=theta, kp=KP, kd=KD, kdd=KDD, tau=TAU)
            T, yout = impulse_response(sys)
            L1 = np.trapz(yout, T)  # ||gamma(t)||_L1

            if L1 >= 1:
                theta_max = theta
            else:
                theta_min = theta
        theta_range[i] = theta_min

    fig = plt.figure()
    plt.plot(h_range, theta_range)
    fig.axes[0].set_ylim(bottom=0, top=0.5)
    fig.axes[0].set_xlim(left=0, right=1.5)
    plt.xlabel("h [s]")
    plt.ylabel("theta_max [s]")
    plt.suptitle("Reproducing Figure 4(b) [Ploeg2014]")
    plt.savefig("lp_string_stability_4b_theta_max.png", dpi=300, deg=False, format="png")
