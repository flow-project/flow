import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from control import bode, feedback, tf, step, pade, bode_plot, tfdata # MATLAB-like functions
from numpy import convolve

# Constants
tau = 0.1 # seconds
kp = 0.2
kd = 0.7
kdd = 0
h = 0.5 # seconds
thetas = [0, 0.15, 0.3]
theta1 = 0.1
theta2 = 1.1

order = 5

# Coefficients in denominator of transfer function
# High order to low order, eg 1*s^2 + 0.1*s + 1
den = [h * tau, tau + h * (1 + kdd), kdd + 1 + h * kd, kd + h * kp, kp]

# non-delayed part of the system
num1 = [kdd, kd, kp]

# delayed part of the system
num_nodelay = [tau, 1, 0, 0]
sys_nodelay = tf(num_nodelay, den)

plt.figure()

print "Attempt 1 of plotting Figure 3(a) from Ploeg2014"
for theta in thetas:
    num_delay, den_delay = pade(theta, n=order)
    sys_delay = tf(num_delay, den_delay)
    sys2 = sys_nodelay * sys_delay
    # num2 = convolve(num_delay, num_nodelay)
    # den2 = convolve(den_delay, den)

    sys1 = tf(num1, den)
    # sys2 = tf(num2, den2)
    # print num2, den2

    sys = sys1 + sys2

    mag, phase, omega = bode_plot(sys, Plot=True)

plt.savefig("lp_string_stability_manual_mag.png", dpi=300, format="png")

print "Attempt 2 of plotting Figure 3(a) from Ploeg2014"
for theta in thetas:
    Hinv = tf([1], [h, 1])
    K = tf([kdd, kd, kp], [1])
    G = tf([1], [tau, 1, 0, 0])
    KG = K * G
    num_delay, den_delay = pade(theta, n=order)
    D = tf(num_delay, den_delay)
    one = tf([1], [1])
    num_KG, den_KG = tfdata(KG + 1)
    invKG1 = tf(den_KG, num_KG)
    sys = invKG1 * (KG + D) * Hinv

    mag, phase, omega = bode_plot(sys, Plot=True)

plt.savefig("lp_string_stability_block_mag.png", dpi=300, format="png")

# FIXME They look the same, so that means the math should be correct,
# but both plots look different from the figure in the paper.
