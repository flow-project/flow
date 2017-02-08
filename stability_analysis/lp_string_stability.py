import numpy as np
from scipy import signal
import scipy
from matplotlib import pyplot as plt
from control import tf, pade, bode_plot, tfdata

import sspade

# Constants
THETA_MIN = 0
THETA_MAX = 1.5

TAU = 0.1  # seconds
KP = 0.2
KD = 0.7
KDD = 0
H = 0.5  # seconds
THETAS = [0, 0.15, 0.3]
HS = [0.1, 0.3, 0.5, 1]

order = 5  # pade approximation order


def construct_delay_ss(n, m, a=0.05):
    """
    :param n: order of the whole system
    :param m: order of the Bessel filter
    :param a: 5% of the delay goes to Bessel, the rest to Pade
    :return:
    """
    peb = sspade.Bessel(m, 1.0 / a)
    Hb = peb.lti_sscascade
    pe = sspade.PadeExponential(n - m, n - m, 1 / (1 - a))
    Hp = pe.lti_sscascade
    H1 = sspade.cascade(Hb, Hp)
    return H1


def construct_system_ss(h=0.0, theta=0.0, kp=0.0, kd=0.0, kdd=0.0, tau=0.0):
    # Coefficients in denominator of transfer function
    # High order to low order, eg 1*s^2 + 0.1*s + 1
    den = [h * tau, tau + h * (1 + kdd), kdd + 1 + h * kd, kd + h * kp, kp]

    # non-delayed part of the system
    num1 = [kdd, kd, kp]
    sys1_ss = signal.StateSpace(signal.lti(num1, den))

    # delayed part of the system, without the delay
    num_nodelay = [tau, 1, 0, 0]
    sys_nodelay_ss = signal.StateSpace(signal.lti(num_nodelay, den))
    # convert to lti form for compatibility with sspade
    sys_nodelay_lti = signal.lti(sys_nodelay_ss.A, sys_nodelay_ss.B,
                                 sys_nodelay_ss.C, sys_nodelay_ss.D)

    # add in the delay to the delayed part of the system, using a mix of
    # bessel and pade approximation in state space form (via sspade)
    delay_ss = construct_delay_ss(100, 10)
    # FIXME how to incorporate theta back into the construction of the delay?
    sys2_ss = sspade.cascade(sys_nodelay_lti, delay_ss)

    # combine the delayed and nondelayed parts
    sys = signal.lti(np.squeeze(scipy.linalg.block_diag(sys1_ss.A, sys2_ss.A)),
                     np.vstack([sys1_ss.B, sys2_ss.B]),
                     np.squeeze(np.hstack([sys1_ss.C, sys2_ss.C])),
                     sys1_ss.D + sys2_ss.D)

    return sys


def construct_system_block(h=0.0, theta=0.0, kp=0.0, kd=0.0, kdd=0.0, tau=0.0):
    """
    See Ploeg2014 for the notation and variable naming used here. "inv"
    indicates inverse.

    :param h:
    :param theta:
    :param kp:
    :param kd:
    :param kdd:
    :param tau:
    :return:
    """
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
    print "Reproducing Figure 3(a) from Ploeg2014 (1): state space " \
          "representation"
    fig = plt.figure()
    for theta in THETAS:
        sys = construct_system_ss(h=H, theta=theta, kp=KP, kd=KD, kdd=KDD,
                                  tau=TAU)
        # mag1, phase1, omega1 = bode_plot(sys, dB=True, deg=False, Plot=True)
        omega1, mag1, phase1 = signal.bode(sys)
        plt.hold(True)
        plt.subplot(211)
        plt.semilogx(omega1, mag1)  # Bode magnitude plot
        plt.subplot(212)
        plt.semilogx(omega1, phase1)  # Bode phase plot

    fig.axes[0].set_ylim(bottom=-5, top=1)
    fig.axes[0].set_xlim(left=0.05, right=10)
    fig.axes[1].set_xlim(left=0.05, right=10)
    plt.suptitle("Reproducing Figure 3(a) [Ploeg2014]")
    plt.savefig("lp_string_stability_3a_ss_mag.png", dpi=300, format="png")

    print "Reproducing Figure 3(a) from Ploeg2014 (2): using block diagram algebra"
    fig = plt.figure()
    for theta in THETAS:
        sys = construct_system_block(h=H, theta=theta, kp=KP, kd=KD, kdd=KDD,
                                     tau=TAU)
        mag2, phase2, omega2 = bode_plot(sys, dB=True, deg=False, Plot=True)

    fig.axes[0].set_ylim(bottom=-5, top=1)
    fig.axes[0].set_xlim(left=0.05, right=10)
    fig.axes[1].set_xlim(left=0.05, right=10)
    plt.suptitle("Reproducing Figure 3(a) [Ploeg2014]")
    plt.savefig("lp_string_stability_3a_block_mag.png", dpi=300, deg=False,
                format="png")

    # assert np.linalg.norm(
    #     mag1 - mag2) < 1e-8, "Magnitudes produced by the two methods are different"
    # assert np.linalg.norm(
    #     omega1 - omega2) < 1e-8, "Frequency list produced by the two methods are different"

    # FIXME why are the phases produced by the two attempts different?
    # assert np.linalg.norm(phase1 - phase2) < 1e-8, "Phases produced by the two methods are different"


    print "Reproducing Figure 3(b) from Ploeg2014"
    h_range = np.linspace(0, 1.5, 20)
    theta_range = np.zeros(h_range.shape)

    for i, h in enumerate(h_range):
        # binary search for theta
        theta_min = THETA_MIN
        theta_max = THETA_MAX
        while abs(theta_min - theta_max) > 1e-3:
            theta = 0.5 * (theta_min + theta_max)
            sys = construct_system_block(h=h, theta=theta, kp=KP, kd=KD,
                                         kdd=KDD, tau=TAU)
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
    plt.savefig("lp_string_stability_3b_theta_max.png", dpi=300, deg=False,
                format="png")

    print "Reproducing Figure 4(a) from Ploeg2014"

    fig = plt.figure()
    for theta in THETAS:
        sys = construct_system_ss(h=H, theta=theta, kp=KP, kd=KD, kdd=KDD,
                                  tau=TAU)
        T, yout = sys.impulse(T=np.linspace(0, 1.5, num=250))
        plt.plot(T, yout, label='theta=%s' % theta)

        L1 = np.trapz(yout, T)  # ||gamma(t)||_L1

    fig.axes[0].set_ylim(bottom=0, top=2.2)
    fig.axes[0].set_xlim(left=0, right=5)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("gamma [m/s^2]")
    plt.suptitle("Reproducing Figure 4(a) [Ploeg2014]")
    plt.savefig("lp_string_stability_4a_impulse.png", dpi=300, deg=False,
                format="png")

    print "Reproducing Figure 4(b) from Ploeg2014"
    h_range = np.linspace(0, 1.5, 20)
    theta_range = np.zeros(h_range.shape)

    for i, h in enumerate(h_range):
        # binary search for theta
        theta_min = THETA_MIN
        theta_max = THETA_MAX
        while abs(theta_min - theta_max) > 1e-3:
            theta = 0.5 * (theta_min + theta_max)
            sys = construct_system_ss(h=h, theta=theta, kp=KP, kd=KD, kdd=KDD,
                                      tau=TAU)
            T, yout = sys.impulse(T=np.linspace(0, 1.5, num=250))
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
    plt.savefig("lp_string_stability_4b_theta_max.png", dpi=300, deg=False,
                format="png")

    print "Now we venture into new territory, exploring alternating communication delays (theta1, theta2)."
    print "That is, for a fixed headway h, what is the admissible space of pairs of communication delays?"
    fig = plt.figure()
    for h in HS:
        theta1_range = np.linspace(THETA_MIN, THETA_MAX, 30)
        theta2_range = np.zeros(theta1_range.shape)

        for i, theta1 in enumerate(theta1_range):
            # binary search for theta
            theta_min = THETA_MIN
            theta_max = THETA_MAX
            while abs(theta_min - theta_max) > 1e-3:
                theta2 = 0.5 * (theta_min + theta_max)
                sys1 = construct_system_block(h=h, theta=theta1, kp=KP, kd=KD,
                                              kdd=KDD, tau=TAU)
                sys2 = construct_system_block(h=h, theta=theta2, kp=KP, kd=KD,
                                              kdd=KDD, tau=TAU)
                sys = sys1 * sys2
                mag, phase, omega = bode_plot(sys, dB=True, Plot=False)

                Hinfty = np.max(mag)  # ||Gamma(s)||_{H_\infty}
                if Hinfty >= 1:
                    theta_max = theta2
                else:
                    theta_min = theta2
            theta2_range[i] = theta_min

        plt.plot(theta2_range, theta1_range, label="h=%s" % h)
        plt.hold(True)

    plt.xlabel("theta1 [s]")
    plt.ylabel("theta2_max [s]")
    plt.legend()
    plt.suptitle(
        "L2 admissible alternating communication delays (theta1, theta2)")
    plt.savefig("lp_string_stability_alternating_comm_delays_L2.png", dpi=300,
                deg=False, format="png")

    print "Same thing now, but with the Linfty condition."
    fig = plt.figure()
    for h in HS:
        theta1_range = np.linspace(THETA_MIN, THETA_MAX, 30)
        theta2_range = np.zeros(theta1_range.shape)

        for i, theta1 in enumerate(theta1_range):
            # binary search for theta
            theta_min = THETA_MIN
            theta_max = THETA_MAX
            while abs(theta_min - theta_max) > 1e-3:
                theta2 = 0.5 * (theta_min + theta_max)
                sys1 = construct_system_ss(h=h, theta=theta1, kp=KP, kd=KD,
                                           kdd=KDD, tau=TAU)
                sys2 = construct_system_ss(h=h, theta=theta2, kp=KP, kd=KD,
                                           kdd=KDD, tau=TAU)
                sys = sspade.cascade(sys1, sys2)
                T, yout = sys.impulse(T=np.linspace(0, 1.5, num=250))
                L1 = np.trapz(yout, T)  # ||gamma(t)||_L1

                if L1 >= 1:
                    theta_max = theta2
                else:
                    theta_min = theta2
            theta2_range[i] = theta_min

        plt.plot(theta2_range, theta1_range, label="h=%s" % h)
        plt.hold(True)

    plt.xlabel("theta1 [s]")
    plt.ylabel("theta2_max [s]")
    plt.legend()
    plt.suptitle(
        "Linfty admissible alternating communication delays (theta1, theta2)")
    plt.savefig("lp_string_stability_alternating_comm_delays_Linfty.png",
                dpi=300, deg=False, format="png")
