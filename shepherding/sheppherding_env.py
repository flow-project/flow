
from flow.envs.loop_accel import SimpleAccelerationEnvironment

class SheppherdingEnv(SimpleAccelerationEnvironment):

    # def __init__(self, env_params, sumo_params, scenario):
        # SimpleAccelerationEnvironment.__init__(self, env_params, sumo_params, scenario)

    def additional_command(self):
        if self.timer % 10 == 0:
            print("aggressive vehicle moving at %+.2f" % self.vehicles.get_speed("aggressive-human_0"))
            print("human vehicle moving at %+.2f" % self.vehicles.get_speed("human_0"))