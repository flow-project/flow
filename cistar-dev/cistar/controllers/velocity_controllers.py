def make_constant_vel_model(speed=8):
    '''
    This function implements a constant velocity control model.
    This is different from most provided control models, which are
    acceleration based.
    :param speed: requested speed for
    :return: function handle for constant velocity control models
    '''
    # k_d = proportional gain
    # k_v = derivative gain
    # s = safe distance

    def constant_vel(carID, env):
        return speed

    return constant_vel
