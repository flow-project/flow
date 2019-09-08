"""Empty init file to ensure documentation for the vehicle class is created."""

from flow.core.kernel.vehicle.base import KernelVehicle
from flow.core.kernel.vehicle.sumo import SumoVehicle
from flow.core.kernel.vehicle.aimsun import AimsunKernelVehicle


__all__ = ['KernelVehicle', 'SumoVehicle', 'AimsunKernelVehicle']
