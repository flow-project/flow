
class KernelPedestrian(object):

    def __init__(self,
            master_kernel):
        self.master_kernel = master_kernel
        self.kernel_api = None

    def pass_api(self, kernel_api):

        self.kernel_api = kernel_api

    def update(self, reset):
        raise NotImplementedError

    # TODO may need to change params
    def add(self, ped_id, edge, pos, lane, speed):
        raise NotImplementedError

    def remove(self, ped_id):
        raise NotImplementedError

    # State acquisition methods

    def get_speed(self, ped_if, error=-1001):
        raise NotImplementedError

    def ged_position(self, ped_id, error=-1001):
        raise NotImplementedError

    def get_edge(self, ped_id, error=""):
        raise NotImplementedError
