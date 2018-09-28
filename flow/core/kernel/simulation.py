class Simulation:
    def __init__(self, kernel):
        self.K = kernel

    def get_speed(self, arg1, arg2):
        return self.K.vehicle.getSpeed(arg1, arg2)

