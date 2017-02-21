def make_constant_vel_model(speed=8):
    # k_d = proportional gain
    # k_v = derivative gain
    # s = safe distance

    def constant_vel(carID, env):
        return speed

    return constant_vel
