"""functions that calculates the expected result for testing."""
import math

# Vehicle Mass
M = 1200
# Gravity
g = 9.81
# Density of Air
ro_air = 1.225
# Rolling resistance coefficient
C_r = .005
# Aerodynamic drag coefficient
C_a = 0.3
# Vehicle Cross sectional Area
A = 2.6
# Road grade
theta = 0


def heavyside(inp):
    """Return 1 if input is positive."""
    return 0 if inp <= 0 else 1


def calculate_power(mu, acceleration, M=M, g=g, theta=theta, C_r=C_r, ro_air=ro_air, A=A, C_a=C_a):
    """Calculate the expected power for POWER_DEMAND_MODEL query."""
    acceleration = (0.8 + ((1 - 0.8) * heavyside(acceleration)) * acceleration)
    accel_and_slope = M * mu * (acceleration + g * math.sin(theta))
    rolling_friction = M * g * C_r * mu
    air_drag = .5 * ro_air * A * C_a * mu**3
    power = accel_and_slope + rolling_friction + air_drag
    return power


def apply_energy_one(row):
    """Apply the power calculation to a row of the dataframe."""
    return [row[0], row[1], calculate_power(row[4], row[6])]